"""Fine-tune Whisper on Khmer speech from OpenSLR SLR42 and Google FLEURS.

Secrets are never written in this file:
- Hugging Face token: see `khmer_asr.hub.login_to_hub`
- Weights & Biases key: the WANDB_API_KEY environment variable (or `wandb login`)
"""

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import evaluate
import torch
import wandb
from datasets import Audio, concatenate_datasets, load_dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from khmer_asr.audio import SAMPLING_RATE, fits_whisper_window, total_duration
from khmer_asr.hub import login_to_hub
from khmer_asr.text import remove_zero_width

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    MODEL_ID:                       str = 'openai/whisper-base'
    MODEL_NAME:                     str = 'whisper-base-aug-10-may-lightning-v1'
    OUT_DIR:                        str = f"./output/{MODEL_NAME}"
    LANGUAGE:                       str = "Khmer"
    TASK:                           str = "transcribe"
    REPORT_TO:                      str = "wandb"
    WANDB_ENTITY:                   str = "KAK-AI"
    WANDB_PROJECT:                  str = "Experiment-Khmer-Whisper"
    BATCH_SIZE:                     int = 16
    EPOCHS:                         int = 5
    GRADIENT_ACCUMULATION_STEPS:    int = 2
    LEARNING_RATE:                float = 5e-5
    SEED:                           int = 42
    VALIDATION_FRACTION:          float = 0.2
    MAX_LABEL_TOKENS:               int = 448   # Whisper decoder limit
    APPLY_SPEC_AUGMENT:            bool = True
    BF16:                          bool = True
    FP16:                          bool = False
    SAVE_TOTAL:           Optional[int] = None


# -------------------- Data --------------------
@dataclass(frozen=True)
class SpeechSource:
    """One Hugging Face speech dataset and where its transcript lives."""
    path: str
    config_name: str
    splits: tuple
    text_column: str


SOURCES = (
    SpeechSource("openslr/openslr", "SLR42", ("train",), text_column="sentence"),
    SpeechSource("google/fleurs", "km_kh", ("train", "validation"), text_column="transcription"),
)


def load_source(source: SpeechSource):
    """Load all splits of `source` as one dataset with only `audio` (16 kHz) and `text`."""
    dataset = concatenate_datasets([
        load_dataset(source.path, source.config_name, split=split, trust_remote_code=True)
        for split in source.splits
    ])
    dataset = dataset.rename_column(source.text_column, "text")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    return dataset.remove_columns([c for c in dataset.column_names if c not in ("audio", "text")])


def load_training_corpus():
    """Every source merged into one dataset, keeping only clips Whisper can hear in full."""
    parts = []
    for source in SOURCES:
        logger.info('⬇️ Downloading %s', source.path)
        dataset = load_source(source)
        logger.info('⏳ %s: %s (%s)', source.path, dataset, total_duration(dataset))
        parts.append(dataset)

    corpus = concatenate_datasets(parts).filter(fits_whisper_window)
    logger.info('⏳ Merged and filtered: %s', corpus)
    return corpus


def make_feature_builder(processor):
    """Return a batched `dataset.map` function: audio -> log-Mel features, text -> label ids."""
    def build_features(batch):
        audio_arrays = [audio['array'] for audio in batch['audio']]
        texts = [remove_zero_width(text) for text in batch['text']]
        return {
            'input_features': processor.feature_extractor(
                audio_arrays, sampling_rate=batch['audio'][0]['sampling_rate']).input_features,
            'labels': processor.tokenizer(
                texts, max_length=TrainingConfig.MAX_LABEL_TOKENS, truncation=True).input_ids,
        }
    return build_features


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads features and labels per batch; padded label positions are ignored by the loss (-100)."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')

        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # The model adds the start token itself; drop it if the tokenizer already did.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch['labels'] = labels
        return batch


# -------------------- Model --------------------
def load_model():
    model = WhisperForConditionalGeneration.from_pretrained(TrainingConfig.MODEL_ID)
    model.generation_config.task = TrainingConfig.TASK
    model.generation_config.forced_decoder_ids = None
    model.config.apply_spec_augment = TrainingConfig.APPLY_SPEC_AUGMENT
    return model


def make_wer_metric(tokenizer):
    """Return a `compute_metrics` function reporting word error rate in percent."""
    wer_metric = evaluate.load('wer')

    def compute_metrics(pred):
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        predictions = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        references = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {'wer': 100 * wer_metric.compute(predictions=predictions, references=references)}
    return compute_metrics


def training_arguments():
    return Seq2SeqTrainingArguments(
        output_dir=TrainingConfig.OUT_DIR,
        per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
        per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
        gradient_accumulation_steps=TrainingConfig.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=TrainingConfig.LEARNING_RATE,
        warmup_steps=1000,
        bf16=TrainingConfig.BF16,
        fp16=TrainingConfig.FP16,
        num_train_epochs=TrainingConfig.EPOCHS,
        eval_strategy='epoch',
        logging_strategy='epoch',
        save_strategy='epoch',
        predict_with_generate=True,
        generation_max_length=256,
        report_to=TrainingConfig.REPORT_TO,
        load_best_model_at_end=True,
        metric_for_best_model='wer',
        greater_is_better=False,
        dataloader_num_workers=6,
        save_total_limit=TrainingConfig.SAVE_TOTAL,
        lr_scheduler_type='constant',
        seed=TrainingConfig.SEED,
        data_seed=TrainingConfig.SEED,
    )


# -------------------- Main --------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
    )
    login_to_hub()
    wandb.login()

    corpus = load_training_corpus()
    split = corpus.train_test_split(test_size=TrainingConfig.VALIDATION_FRACTION,
                                    seed=TrainingConfig.SEED, shuffle=True)
    train_set, validate_set = split['train'], split['test']

    run = wandb.init(
        entity=TrainingConfig.WANDB_ENTITY,
        project=TrainingConfig.WANDB_PROJECT,
        name=TrainingConfig.MODEL_NAME,
        config={
            "learning_rate": TrainingConfig.LEARNING_RATE,
            "model": TrainingConfig.MODEL_ID,
            "epochs": TrainingConfig.EPOCHS,
            "lang": TrainingConfig.LANGUAGE,
            "batch_size": TrainingConfig.BATCH_SIZE,
            "dataset": [source.path for source in SOURCES],
            "total_num": corpus.num_rows,
            "total_hour": total_duration(corpus),
            "train_set": train_set.num_rows,
            "test_set": validate_set.num_rows,
            "train_hour": total_duration(train_set),
            "test_hour": total_duration(validate_set),
        },
    )

    logger.info('⬇️ Preprocessing dataset')
    processor = WhisperProcessor.from_pretrained(TrainingConfig.MODEL_ID,
                                                 language=TrainingConfig.LANGUAGE, task=TrainingConfig.TASK)
    build_features = make_feature_builder(processor)
    train_set = train_set.map(build_features, remove_columns=train_set.column_names, num_proc=4, batched=True)
    validate_set = validate_set.map(build_features, remove_columns=validate_set.column_names, num_proc=4, batched=True)

    logger.info('⬇️ Loading model')
    model = load_model()
    trainer = Seq2SeqTrainer(
        args=training_arguments(),
        model=model,
        train_dataset=train_set,
        eval_dataset=validate_set,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        ),
        compute_metrics=make_wer_metric(processor.tokenizer),
        tokenizer=processor.feature_extractor,
    )

    logger.info('⬇️ Training model')
    trainer.train()
    processor.save_pretrained(TrainingConfig.OUT_DIR)
    run.finish()

    logger.info('✅ Push to HuggingFace')
    trainer.push_to_hub(commit_message="Upload checkpoint")


if __name__ == "__main__":
    main()

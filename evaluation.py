"""Transcribe an evaluation set with one model and push the transcriptions to the Hub.

The transcriptions are added as a new column named after the model, so
compute_cer.py can later score every model stored in the same dataset.

    !pip install datasets
"""

from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from khmer_asr.audio import SAMPLING_RATE
from khmer_asr.hub import login_to_hub


@dataclass(frozen=True)
class Config:
    MODEL_NAME:            str = "PhanithLIM/whisper-medium-aug-05-june"
    PROCESSOR_NAME:        str = "PhanithLIM/whisper-medium-aug-05-june"
    TASK:                  str = "transcribe"
    LANGUAGE:              str = "Khmer"
    DATASET_NAME:          str = "PhanithLIM/asr-wmc-evaluate"
    PREDICTION_COLUMN:     str = MODEL_NAME.split("/")[-1]
    SPLIT:                 str = "test"
    DEVICE:                str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_LENGTH:  Optional[int] = 2024
    NUM_BEAMS:   Optional[int] = 5


def make_transcriber(model, processor):
    """Return a `dataset.map` function that adds the model's transcription to a row."""
    def transcribe(example):
        audio = example["audio"]
        features = processor(audio["array"], sampling_rate=audio["sampling_rate"],
                             return_tensors="pt").input_features
        with torch.no_grad():
            predicted_ids = model.generate(features.to(Config.DEVICE), language=Config.LANGUAGE,
                                           max_length=Config.MAX_LENGTH, num_beams=Config.NUM_BEAMS)[0]
        example[Config.PREDICTION_COLUMN] = processor.decode(predicted_ids, skip_special_tokens=True)
        return example
    return transcribe


def main():
    login_to_hub()

    dataset = load_dataset(Config.DATASET_NAME, split=Config.SPLIT)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    print(dataset)

    model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_NAME).to(Config.DEVICE)
    processor = WhisperProcessor.from_pretrained(Config.PROCESSOR_NAME,
                                                 language=Config.LANGUAGE, task=Config.TASK)

    result = dataset.map(make_transcriber(model, processor))
    result.push_to_hub(Config.DATASET_NAME, split=Config.SPLIT)
    print("Results pushed to the hub successfully.")


if __name__ == "__main__":
    main()

"""Character error rate (CER) of every model on a Khmer evaluation set.

The dataset has an `audio` column, a reference `text` column, and one column
of transcriptions per model (evaluation.py adds those). Prints one CER per model.

    !pip install python-Levenshtein tha datasets khmerspell prettytable
"""

from dataclasses import dataclass

from datasets import load_dataset
from Levenshtein import distance
from prettytable import PrettyTable

from khmer_asr.audio import fits_whisper_window
from khmer_asr.hub import login_to_hub
from khmer_asr.text import has_latin, normalize_for_scoring


@dataclass(frozen=True)
class Config:
    DATASET_NAME:            str = 'PhanithLIM/gfleurs-evaluation'
    SPLIT:                   str = 'test'
    AUDIO_COLUMN:            str = 'audio'
    TEXT_COLUMN:             str = 'text'
    # An utterance this far from its reference is a broken prediction
    # (e.g. a hallucination loop), not a recognition error; it is left out.
    MAX_EDITS_PER_UTTERANCE: int = 100


def corpus_cer(references, hypotheses, max_edits: int) -> float:
    """CER in percent over all (reference, hypothesis) pairs, pooled by character.

    Pairs where either side is blank, or whose edit distance reaches
    `max_edits`, are skipped. Returns 0.0 if nothing is left to score.
    """
    total_chars = total_edits = 0
    for reference, hypothesis in zip(references, hypotheses):
        if not (reference or "").strip() or not (hypothesis or "").strip():
            continue
        edits = distance(reference, hypothesis)
        if edits < max_edits:
            total_chars += len(reference)
            total_edits += edits
    return 100 * total_edits / total_chars if total_chars else 0.0


def _normalize_text_columns(example):
    return {column: normalize_for_scoring(value)
            for column, value in example.items() if isinstance(value, str)}


def _is_scoreable(example) -> bool:
    """Keep rows with some text and no Latin script in any column."""
    texts = [value for value in example.values() if isinstance(value, str)]
    return any(text.strip() for text in texts) and not any(has_latin(text) for text in texts)


def main():
    login_to_hub()

    dataset = load_dataset(Config.DATASET_NAME, split=Config.SPLIT)
    print(dataset)
    dataset = (dataset
               .filter(fits_whisper_window)
               .map(_normalize_text_columns)
               .filter(_is_scoreable))

    table = PrettyTable(field_names=['model', 'cer (%)'])
    table.align = 'l'
    references = dataset[Config.TEXT_COLUMN]
    model_columns = [c for c in dataset.column_names
                     if c not in (Config.AUDIO_COLUMN, Config.TEXT_COLUMN)]
    for column in model_columns:
        cer = corpus_cer(references, dataset[column], Config.MAX_EDITS_PER_UTTERANCE)
        table.add_row([column, f"{cer:.3f}"])

    print('Model Performance on Khmer Test Set')
    print(table)


if __name__ == "__main__":
    main()

"""Audio rules shared by every script that touches Hugging Face audio datasets.

An "example" is one dataset row with an `audio` column holding the decoded
`array` and its `sampling_rate`.
"""

SAMPLING_RATE = 16_000
"""Whisper is trained on 16 kHz audio; every dataset is resampled to this."""

WHISPER_WINDOW_SECONDS = 30
"""Whisper sees at most 30 s of audio at once; longer clips are cut off."""


def duration_seconds(example) -> float:
    audio = example["audio"]
    return len(audio["array"]) / audio["sampling_rate"]


def fits_whisper_window(example) -> bool:
    """True if Whisper can hear the whole clip. Use as a `dataset.filter` predicate."""
    return duration_seconds(example) <= WHISPER_WINDOW_SECONDS


def total_duration(dataset) -> str:
    """Total audio length of a dataset, formatted like "12h 03m 09s"."""
    seconds = int(sum(duration_seconds(example) for example in dataset))
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m {seconds % 60:02d}s"

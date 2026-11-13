"""Khmer text normalization shared by training, evaluation and alignment.

Every rule about what Khmer text should look like ("numbers are spoken as
words", "punctuation is not scored", ...) lives here, so the scripts cannot
drift apart. Heavy third-party packages are imported only by the functions
that need them, so importing this module is always cheap.
"""

import re
from functools import cache

ZERO_WIDTH_SPACE = "​"

# Digits (Arabic or Khmer), optional thousands separators, optional decimals.
# Commas count only *between* digits, so ordinary punctuation is left alone.
_NUMBER = re.compile(r"[0-9០-៩]+(?:,[0-9០-៩]+)*(?:\.[0-9០-៩]+)?")
_LATIN = re.compile(r"[a-zA-Z]")
_NOT_ROMAN = re.compile(r"[^a-z']")

# Characters that carry no pronunciation, so they are ignored when scoring.
_UNSPOKEN_SYMBOLS = (
    "(", ")", "[", "]", "{", "}", "<", ">",
    "“", "”", "‘", "’", "«", "»", ",", "?",
    "「", "」", "『", "』", "▁", "-", " ", "%", ".",
    "៖", "។", "៛", "៕", "!", ZERO_WIDTH_SPACE, "–", "ៗ", "�",
)


def remove_zero_width(text: str) -> str:
    """Remove zero-width spaces, which Khmer editors insert as invisible word breaks."""
    return text.replace(ZERO_WIDTH_SPACE, "")


def spell_out_numbers(text: str) -> str:
    """Replace every number (Arabic or Khmer digits) with its Khmer words, e.g. "12" -> "ដប់ពីរ"."""
    from tha.decimals import processor

    return _NUMBER.sub(lambda m: processor(m.group(0)).replace("▁", " "), text)


def has_latin(text: str) -> bool:
    """True if the text contains any Latin letter."""
    return _LATIN.search(text) is not None


def normalize_for_scoring(text: str) -> str:
    """Reduce text to the spoken characters only, in canonical Khmer order.

    Two transcriptions that sound the same normalize to the same string, so an
    edit distance between normalized strings counts only real recognition errors.
    """
    from khmerspell import khnormal

    text = spell_out_numbers(text)
    for symbol in _UNSPOKEN_SYMBOLS:
        text = text.replace(symbol, "")
    return khnormal(text)


def romanize(text: str) -> str:
    """Romanize Khmer text into the lowercase a-z/' alphabet the MMS aligner understands.

    Numbers are spelled out first and all spaces are dropped, so the result is
    one continuous token string per input line.
    """
    romanized = _uroman().romanize_string(spell_out_numbers(text), lcode="khm")
    return _NOT_ROMAN.sub("", romanized.lower().replace("’", "'"))


@cache
def _uroman():
    import uroman

    return uroman.Uroman()

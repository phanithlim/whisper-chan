"""Split a Khmer text file into numbered sentences (one transcript line per sentence).

    uv run --extra align automate_sementation.py data/8.txt
"""

import argparse
from typing import List

from khmernltk import sentence_tokenize
from khmerspell import khnormal


def split_sentences(path: str) -> List[str]:
    """Sentences of the whole file, after joining its non-blank lines and normalizing."""
    with open(path, encoding='utf-8') as f:
        text = ' '.join(line.strip() for line in f if line.strip())
    return sentence_tokenize(khnormal(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", default="data/8.txt")
    for i, sentence in enumerate(split_sentences(parser.parse_args().path), start=1):
        print(f"{i:02d}: {sentence}")

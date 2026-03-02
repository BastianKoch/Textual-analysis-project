"""
normalize_corpora.py
--------------------
Apply the standard NLP normalization pipeline to every E/S/G corpus text file:

    1. Lowercase
    2. Remove punctuation
    3. Remove digits
    4. Remove English stopwords (NLTK)
    5. Tokenize by whitespace → single space-separated flat string

Input:  data/processed/Text corpus/{E,S,G}/*.txt
Output: same paths but under data/processed/Text corpus cleaned/{E,S,G}/

The originals in 'Text corpus/' are never modified.
"""

from __future__ import annotations

import math
import re
import string
from pathlib import Path

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
_STOPWORDS   = set(stopwords.words("english"))
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
IN_ROOT      = DATA_DIR / "Text corpus"
OUT_ROOT     = DATA_DIR / "Text corpus cleaned"


def normalize(text: str) -> str:
    """Lowercase → strip punctuation → strip digits → remove stopwords → join."""
    tokens = []
    for line in text.splitlines():
        line = line.lower()
        line = line.translate(_PUNCT_TABLE)      # remove punctuation
        line = re.sub(r"\d+", "", line)           # remove digits
        for tok in line.split():
            if tok and tok not in _STOPWORDS:
                tokens.append(tok)
    return " ".join(tokens)


# Discover all .txt files under Text corpus/
corpus_files = sorted(IN_ROOT.rglob("*.txt"))

if not corpus_files:
    raise FileNotFoundError(f"No .txt files found under {IN_ROOT}")

for src in corpus_files:
    # Reconstruct output path, preserving E/S/G sub-directory
    rel = src.relative_to(IN_ROOT)
    dst = OUT_ROOT / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw  = src.read_text(encoding="utf-8")
    norm = normalize(raw)
    dst.write_text(norm, encoding="utf-8")

    in_words  = len(raw.split())
    out_words = len(norm.split())
    print(f"{rel}:  {in_words:,} → {out_words:,} tokens  (saved → {dst})")

print("\nDone.")

"""
build_tfidf.py
--------------
Build a bigram TF-IDF matrix over all earnings-call transcripts.

Formula
-------
    tfidf(b, t) = f(b, t) * log(T / df(b))

where
    f(b, t)  = raw frequency of bigram b in transcript t
               (presentation + answers + questions segments combined)
    T        = total number of transcripts
    df(b)    = number of transcripts that contain bigram b  (≥1 occurrence)

Outputs  (all in data/processed/)
-------
    bigram_tfidf.npz          — scipy CSR sparse matrix  (T × V)
    bigram_vocab.csv          — vocabulary: index, bigram
    bigram_transcript_index.csv — row index: row, transcript_id
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
BIGRAM_DIR   = DATA_DIR / "interim" / "bigrams"
OUT_DIR      = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Discover transcripts and their segment files
# ---------------------------------------------------------------------------
_SUFFIXES = ("_presentation.txt", "_answers.txt", "_questions.txt")

def transcript_id(filename: str) -> str | None:
    for sfx in _SUFFIXES:
        if filename.endswith(sfx):
            return filename[: -len(sfx)]
    return None

# group files by transcript id
transcript_files: dict[str, list[Path]] = defaultdict(list)
for f in sorted(BIGRAM_DIR.iterdir()):
    tid = transcript_id(f.name)
    if tid:
        transcript_files[tid].append(f)

transcript_ids = sorted(transcript_files.keys())
T = len(transcript_ids)
log.info("Found %d transcripts", T)

# ---------------------------------------------------------------------------
# 2. First pass — build per-transcript Counters and document frequencies
# ---------------------------------------------------------------------------
log.info("Pass 1: counting bigrams per transcript …")

tf: list[Counter] = []          # tf[i] = Counter of bigrams for transcript i
df: Counter = Counter()         # df[bigram] = # transcripts containing it

for i, tid in enumerate(transcript_ids):
    c: Counter = Counter()
    for fpath in transcript_files[tid]:
        text = fpath.read_text(encoding="utf-8").strip()
        if text:
            c.update(text.splitlines())   # each line is one "word1 word2" bigram
    tf.append(c)
    df.update(c.keys())                   # increment df for every present bigram

    if (i + 1) % 5000 == 0 or (i + 1) == T:
        log.info("  %d / %d transcripts processed", i + 1, T)

# ---------------------------------------------------------------------------
# 3. Build vocabulary (bigrams that appear in ≥1 transcript)
# ---------------------------------------------------------------------------
log.info("Building vocabulary …")
vocab_list = sorted(df.keys())
V = len(vocab_list)
bigram_to_idx = {b: j for j, b in enumerate(vocab_list)}
log.info("Vocabulary size: %d bigrams", V)

# ---------------------------------------------------------------------------
# 4. Compute TF-IDF and build COO sparse matrix
# ---------------------------------------------------------------------------
log.info("Pass 2: computing TF-IDF and building sparse matrix …")

rows, cols, vals = [], [], []

for i, c in enumerate(tf):
    for bigram, freq in c.items():
        j = bigram_to_idx[bigram]
        tfidf_val = freq * math.log(T / df[bigram])
        rows.append(i)
        cols.append(j)
        vals.append(tfidf_val)

    if (i + 1) % 5000 == 0 or (i + 1) == T:
        log.info("  %d / %d rows built", i + 1, T)

matrix = sp.csr_matrix(
    (np.array(vals, dtype=np.float32),
     (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
    shape=(T, V),
)
log.info("Matrix shape: %s  nnz: %d  density: %.6f%%",
         matrix.shape, matrix.nnz, 100 * matrix.nnz / (T * V))

# ---------------------------------------------------------------------------
# 5. Save outputs
# ---------------------------------------------------------------------------
# sparse matrix
mat_path = OUT_DIR / "bigram_tfidf.npz"
sp.save_npz(str(mat_path), matrix)
log.info("Saved sparse matrix → %s", mat_path)

# vocabulary
vocab_path = OUT_DIR / "bigram_vocab.csv"
with open(vocab_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["index", "bigram"])
    for j, b in enumerate(vocab_list):
        w.writerow([j, b])
log.info("Saved vocabulary (%d rows) → %s", V, vocab_path)

# transcript index
idx_path = OUT_DIR / "bigram_transcript_index.csv"
with open(idx_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["row", "transcript_id"])
    for i, tid in enumerate(transcript_ids):
        w.writerow([i, tid])
log.info("Saved transcript index (%d rows) → %s", T, idx_path)

log.info("All done.")

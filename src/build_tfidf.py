"""
build_tfidf.py
--------------
Build three bigram TF-IDF sparse matrices over all earnings-call transcripts:

    combined   — presentation + answers segments merged per transcript
    pres       — presentation segment only
    answers    — answers segment only

Formula
-------
    tfidf(b, t) = f(b, t) * log(T / df(b))

where
    f(b, t)  = raw frequency of bigram b in transcript t (for the given set)
    T        = total number of transcripts
    df(b)    = number of transcripts that contain bigram b in the given set

Each set produces three files in data/processed/:
    bigram_tfidf_{label}.npz            scipy CSR float32  (T × V_label)
    bigram_vocab_{label}.csv            index, bigram
    bigram_transcript_index.csv         row, transcript_id  (shared, written once)
"""

from __future__ import annotations

import csv
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
# Segment suffix groups to process
# ---------------------------------------------------------------------------
SETS: dict[str, tuple[str, ...]] = {
    "combined": ("_presentation.txt", "_answers.txt"),
    "pres":     ("_presentation.txt",),
    "answers":  ("_answers.txt",),
}

_ALL_SUFFIXES = ("_presentation.txt", "_answers.txt", "_questions.txt")

def _transcript_id(filename: str) -> str | None:
    for sfx in _ALL_SUFFIXES:
        if filename.endswith(sfx):
            return filename[: -len(sfx)]
    return None

# ---------------------------------------------------------------------------
# Discover all transcript IDs and map each to its available segment files
# ---------------------------------------------------------------------------
all_segment_files: dict[str, dict[str, Path]] = defaultdict(dict)
for f in sorted(BIGRAM_DIR.iterdir()):
    tid = _transcript_id(f.name)
    if tid is None:
        continue
    for sfx in _ALL_SUFFIXES:
        if f.name.endswith(sfx):
            all_segment_files[tid][sfx] = f

transcript_ids = sorted(all_segment_files.keys())
T = len(transcript_ids)
log.info("Found %d transcripts", T)

# ---------------------------------------------------------------------------
# Write shared transcript index (row → transcript_id)
# ---------------------------------------------------------------------------
idx_path = OUT_DIR / "bigram_transcript_index.csv"
with open(idx_path, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["row", "transcript_id"])
    for i, tid in enumerate(transcript_ids):
        w.writerow([i, tid])
log.info("Saved transcript index → %s", idx_path)


# ---------------------------------------------------------------------------
# Helper: build and save one TF-IDF set
# ---------------------------------------------------------------------------
def build_set(label: str, suffixes: tuple[str, ...]) -> None:
    log.info("=== Building set: %s  (segments: %s) ===", label, suffixes)

    # Pass 1 — per-transcript Counters + document frequencies
    log.info("  Pass 1: counting bigrams …")
    tf: list[Counter] = []
    df: Counter = Counter()

    for i, tid in enumerate(transcript_ids):
        c: Counter = Counter()
        seg_map = all_segment_files[tid]
        for sfx in suffixes:
            fpath = seg_map.get(sfx)
            if fpath is None:
                continue
            text = fpath.read_text(encoding="utf-8").strip()
            if text:
                c.update(text.splitlines())
        tf.append(c)
        df.update(c.keys())

        if (i + 1) % 5000 == 0 or (i + 1) == T:
            log.info("    %d / %d transcripts", i + 1, T)

    # Build vocabulary
    vocab_list = sorted(df.keys())
    V = len(vocab_list)
    bigram_to_idx = {b: j for j, b in enumerate(vocab_list)}
    log.info("  Vocabulary: %d bigrams", V)

    # Pass 2 — TF-IDF → sparse matrix
    log.info("  Pass 2: computing TF-IDF …")
    rows, cols, vals = [], [], []

    for i, c in enumerate(tf):
        for bigram, freq in c.items():
            j = bigram_to_idx[bigram]
            rows.append(i)
            cols.append(j)
            vals.append(freq * math.log(T / df[bigram]))

        if (i + 1) % 5000 == 0 or (i + 1) == T:
            log.info("    %d / %d rows", i + 1, T)

    matrix = sp.csr_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(T, V),
    )
    log.info("  Matrix shape: %s  nnz: %d  density: %.6f%%",
             matrix.shape, matrix.nnz, 100 * matrix.nnz / (T * V))

    # Save sparse matrix
    mat_path = OUT_DIR / f"bigram_tfidf_{label}.npz"
    sp.save_npz(str(mat_path), matrix)
    log.info("  Saved matrix    → %s", mat_path)

    # Save vocabulary
    vocab_path = OUT_DIR / f"bigram_vocab_{label}.csv"
    with open(vocab_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "bigram"])
        for j, b in enumerate(vocab_list):
            w.writerow([j, b])
    log.info("  Saved vocab     → %s", vocab_path)


# ---------------------------------------------------------------------------
# Run all three sets
# ---------------------------------------------------------------------------
for label, suffixes in SETS.items():
    build_set(label, suffixes)

log.info("All done.")

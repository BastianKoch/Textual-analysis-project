"""
build_corpus_esg_vectors.py
---------------------------
Project each ESG reference corpus (E, S, G) into the bigram TF-IDF space
that was built from the earnings-call transcripts.

Formula
-------
For each bigram b in corpus c:
    tfidf(b, c) = freq(b, c) * log(T / df_transcript(b))

where
    freq(b, c)           = raw count of bigram b in corpus c
    T                    = number of transcripts (from the transcript matrix)
    df_transcript(b)     = number of transcripts containing bigram b
                           (recovered as column-wise nnz of the transcript matrix)

Inputs
------
    data/processed/Text corpus cleaned/{E,S,G}/*.txt
    data/processed/bigram_tfidf_{label}.npz     (one per set)
    data/processed/bigram_vocab_{label}.csv     (one per set)

Outputs  (data/processed/esg_corpus_vectors/)
-------
    corpus_tfidf_{label}_{E|S|G}.npz   — scipy CSR sparse vector  (1 × V)
    corpus_stats_{label}.csv           — summary: corpus, bigrams_in_vocab,
                                         bigrams_oov, total_tfidf_mass
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
CORPUS_ROOT  = DATA_DIR / "processed" / "Text corpus cleaned"
PROC_DIR     = DATA_DIR / "processed"
OUT_DIR      = DATA_DIR / "processed" / "esg_corpus_vectors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Which sets to process
# ---------------------------------------------------------------------------
SETS = ("combined", "pres", "answers")

# ---------------------------------------------------------------------------
# Corpus files
# ---------------------------------------------------------------------------
CORPORA: dict[str, list[Path]] = {
    label: sorted((CORPUS_ROOT / label_dir).rglob("*.txt"))
    for label, label_dir in [("E", "E"), ("S", "S"), ("G", "G")]
}


def read_corpus(paths: list[Path]) -> str:
    return " ".join(p.read_text(encoding="utf-8") for p in paths)


def bigram_counter(text: str) -> Counter:
    tokens = text.split()
    return Counter(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
for set_label in SETS:
    mat_path   = PROC_DIR / f"bigram_tfidf_{set_label}.npz"
    vocab_path = PROC_DIR / f"bigram_vocab_{set_label}.csv"

    if not mat_path.exists() or not vocab_path.exists():
        print(f"[SKIP] Missing files for set '{set_label}' — run build_tfidf.py first.")
        continue

    print(f"\n=== Set: {set_label} ===")

    # Load vocab: bigram → column index
    print("  Loading vocabulary …")
    bigram_to_idx: dict[str, int] = {}
    with open(vocab_path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            bigram_to_idx[row[1]] = int(row[0])
    V = len(bigram_to_idx)
    print(f"  Vocabulary size: {V:,}")

    # Load sparse matrix and recover df from column nnz
    print("  Loading transcript TF-IDF matrix …")
    mat = sp.load_npz(str(mat_path))
    T   = mat.shape[0]

    # column-wise nnz = number of transcripts containing that bigram
    print("  Computing document frequencies from matrix …")
    col_nnz = np.diff(mat.tocsc().indptr)   # fast CSC column nnz
    # col_nnz[j] = df for bigram at vocab index j

    stats_rows = []

    for corpus_label, corpus_paths in CORPORA.items():
        if not corpus_paths:
            print(f"  [WARN] No files found for corpus {corpus_label}, skipping.")
            continue

        text = read_corpus(corpus_paths)
        bg_counts = bigram_counter(text)

        rows_coo, cols_coo, vals_coo = [], [], []
        in_vocab = 0
        oov      = 0

        for bigram, freq in bg_counts.items():
            j = bigram_to_idx.get(bigram)
            if j is None:
                oov += 1
                continue
            df_b = int(col_nnz[j])
            if df_b == 0:
                oov += 1
                continue
            in_vocab += 1
            tfidf_val = freq * math.log(T / df_b)
            rows_coo.append(0)
            cols_coo.append(j)
            vals_coo.append(tfidf_val)

        vec = sp.csr_matrix(
            (np.array(vals_coo, dtype=np.float32),
             (np.array(rows_coo, dtype=np.int32),
              np.array(cols_coo, dtype=np.int32))),
            shape=(1, V),
        )

        out_path = OUT_DIR / f"corpus_tfidf_{set_label}_{corpus_label}.npz"
        sp.save_npz(str(out_path), vec)

        total_mass = float(np.sum(vals_coo))
        stats_rows.append({
            "corpus":           corpus_label,
            "set":              set_label,
            "bigrams_in_vocab": in_vocab,
            "bigrams_oov":      oov,
            "total_tfidf_mass": round(total_mass, 2),
        })
        print(f"  [{corpus_label}] in_vocab={in_vocab:,}  oov={oov:,}  "
              f"tfidf_mass={total_mass:,.1f}  → {out_path.name}")

    # Save stats
    stats_path = OUT_DIR / f"corpus_stats_{set_label}.csv"
    with open(stats_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["corpus", "set", "bigrams_in_vocab",
                                           "bigrams_oov", "total_tfidf_mass"])
        w.writeheader()
        w.writerows(stats_rows)
    print(f"  Stats → {stats_path.name}")

print("\nAll done.")

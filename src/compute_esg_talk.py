"""
compute_esg_talk.py
-------------------
Compute per-transcript ESG cosine-similarity scores (e_talk, s_talk, g_talk)
for each of the three segment sets: combined, pres, answers.

For every transcript t and corpus c ∈ {E, S, G}:

    talk(t, c) = cos( tfidf_transcript(t),  tfidf_corpus(c) )
               = (t · c) / (‖t‖ · ‖c‖)

where both vectors live in the same bigram TF-IDF space.

Inputs
------
    data/processed/bigram_tfidf_{combined,pres,answers}.npz
    data/processed/esg_corpus_vectors/corpus_tfidf_{set}_{E,S,G}.npz
    data/processed/bigram_transcript_index.csv

Output
------
    data/processed/esg_talk.csv
    Columns:
        transcript_id
        e_talk_combined,  s_talk_combined,  g_talk_combined
        e_talk_pres,      s_talk_pres,      g_talk_pres
        e_talk_answers,   s_talk_answers,   g_talk_answers
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR     = PROJECT_ROOT / "data" / "processed"
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
VECTORS_DIR  = PROC_DIR / "esg_corpus_vectors"

SETS    = ("combined", "pres", "answers")
CORPORA = ("E", "S", "G")

# ---------------------------------------------------------------------------
# Load transcript row index (maps matrix row → transcript_id)
# ---------------------------------------------------------------------------
index_path = PROC_DIR / "bigram_transcript_index.csv"
with open(index_path, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    rows = sorted(reader, key=lambda r: int(r["row"]))
transcript_ids = [r["transcript_id"] for r in rows]
T = len(transcript_ids)
print(f"Transcripts: {T:,}")

# ---------------------------------------------------------------------------
# Load call metadata (firm identifiers + date)
# ---------------------------------------------------------------------------
META_COLS = ["permco", "permno", "gvkey", "comnam",
             "filename", "date_call", "year_call", "month_call", "day_call"]
df_meta = pd.read_csv(RAW_DIR / "list_earnings_calls_group_project_upload.csv",
                      dtype=str)
df_meta["transcript_id"] = df_meta["filename"].str.replace(r"\.txt$", "",
                                                             regex=True)
meta_lookup: dict[str, dict] = (
    df_meta.set_index("transcript_id")[META_COLS].to_dict(orient="index")
)
print(f"Metadata rows: {len(meta_lookup):,}")

# ---------------------------------------------------------------------------
# Compute cosine similarities
# ---------------------------------------------------------------------------
# results[transcript_id] = dict of {column_name: value}
results: dict[str, dict[str, float]] = {tid: {} for tid in transcript_ids}

for set_label in SETS:
    print(f"\nSet: {set_label}")

    # Load transcript matrix and L2-normalise rows
    mat = sp.load_npz(str(PROC_DIR / f"bigram_tfidf_{set_label}.npz"))
    print(f"  Matrix shape: {mat.shape}")

    mat_norm = normalize(mat, norm="l2", copy=True)   # (T, V) row-normalised

    for corpus_label in CORPORA:
        vec_path = VECTORS_DIR / f"corpus_tfidf_{set_label}_{corpus_label}.npz"
        corpus_vec = sp.load_npz(str(vec_path))       # (1, V) sparse

        # L2-normalise the corpus vector
        corpus_norm = normalize(corpus_vec, norm="l2", copy=True)  # (1, V)

        # Cosine similarity = dot product of L2-normalised vectors
        # mat_norm (T, V)  ×  corpus_norm.T (V, 1)  →  (T, 1) dense
        sims = mat_norm.dot(corpus_norm.T).toarray().ravel()   # (T,)

        col_name = f"{corpus_label.lower()}_talk_{set_label}"
        for tid, sim in zip(transcript_ids, sims):
            results[tid][col_name] = float(sim)

        mean_sim = float(np.mean(sims))
        nonzero  = int(np.sum(sims > 0))
        print(f"  [{corpus_label}] mean={mean_sim:.6f}  non-zero={nonzero:,}")

# ---------------------------------------------------------------------------
# Write output CSV
# ---------------------------------------------------------------------------
col_order = [
    f"{c.lower()}_talk_{s}"
    for s in SETS
    for c in CORPORA
]

out_path = PROC_DIR / "esg_talk.csv"
with open(out_path, "w", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    writer.writerow(META_COLS + col_order)
    for tid in transcript_ids:
        meta = meta_lookup.get(tid, {c: "" for c in META_COLS})
        meta_vals = [meta.get(c, "") for c in META_COLS]
        score_vals = [results[tid].get(c, 0.0) for c in col_order]
        writer.writerow(meta_vals + score_vals)

print(f"\nSaved → {out_path}  ({T:,} rows × {len(META_COLS) + len(col_order)} columns)")

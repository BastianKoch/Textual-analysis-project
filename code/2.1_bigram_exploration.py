"""
Bigram Exploration — Top-15 per ESG Corpus
===========================================

Plot the top-15 bigrams per ESG corpus ranked by TF-IDF weight.

Input
-----
    data/processed/bigram_vocab_{SEGMENT}.csv
    data/processed/bigram_tfidf_{SEGMENT}.npz
    data/processed/esg_corpus_vectors/corpus_tfidf_{SEGMENT}_{E|S|G}.npz

Output
------
    output/figures/esg_top_bigrams_{SEGMENT}.png

Usage
-----
    python code/2.1_bigram_exploration.py

Authors
-------
    Bastian Koch
"""

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns

# ── paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[1]
PROC_DIR    = ROOT / "data" / "processed"
VECTORS_DIR = PROC_DIR / "esg_corpus_vectors"
FIGURES_DIR = ROOT / "output" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── configuration ─────────────────────────────────────────────────────────────
SEGMENT = "combined"   # "combined" | "pres" | "answers"
CHART_N = 15           # bigrams per panel

CORPORA       = ("E", "S", "G")
CORPUS_LABELS = {"E": "Environment", "S": "Social", "G": "Governance"}

# ── load vocab ────────────────────────────────────────────────────────────────
vocab_path = PROC_DIR / f"bigram_vocab_{SEGMENT}.csv"
print(f"Loading vocab from {vocab_path.name} …", flush=True)

with open(vocab_path, newline="", encoding="utf-8") as fh:
    reader = csv.reader(fh)
    next(reader)
    rows_v = list(reader)

idx_to_bigram = [""] * len(rows_v)
for row in rows_v:
    idx_to_bigram[int(row[0])] = row[1]

print(f"  Vocabulary size : {len(idx_to_bigram):,}")

# ── load transcript matrix → recover df and T ─────────────────────────────────
mat_path = PROC_DIR / f"bigram_tfidf_{SEGMENT}.npz"
print(f"Loading transcript matrix from {mat_path.name} …", flush=True)
mat     = sp.load_npz(str(mat_path))
T       = mat.shape[0]
col_nnz = np.diff(mat.tocsc().indptr)   # document frequency per bigram
print(f"  Transcripts     : {T:,}")
del mat


# ── helper ────────────────────────────────────────────────────────────────────
def load_top_n(corpus_label: str, n: int) -> pd.DataFrame:
    """Load corpus TF-IDF vector and return the top-N bigrams."""
    vec_path = VECTORS_DIR / f"corpus_tfidf_{SEGMENT}_{corpus_label}.npz"
    vec      = sp.load_npz(str(vec_path)).tocsr()   # (1, V)

    nz_cols = vec.indices
    nz_vals = vec.data

    order    = np.argsort(nz_vals)[::-1][:n]
    rows_out = []
    for o in order:
        j       = int(nz_cols[o])
        df_b    = int(col_nnz[j])
        idf_b   = math.log(T / df_b)
        tfidf_b = float(nz_vals[o])
        rows_out.append({
            "bigram": idx_to_bigram[j],
            "tf":     round(tfidf_b / idf_b),
            "idf":    idf_b,
            "tf-idf": tfidf_b,
        })

    df   = pd.DataFrame(rows_out)
    tmin = df["tf-idf"].min()
    tmax = df["tf-idf"].max()
    df["tf-idf (norm)"] = (df["tf-idf"] - tmin) / (tmax - tmin) if tmax > tmin else 0.0
    return df


# ── plot ──────────────────────────────────────────────────────────────────────
_pal   = sns.color_palette("crest", 3)
COLORS = {"E": _pal[0], "S": _pal[1], "G": _pal[2]}

mpl.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.grid":          False,
})

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, corp in zip(axes, CORPORA):
    df_plot = load_top_n(corp, CHART_N).sort_values("tf-idf")
    ax.barh(df_plot["bigram"], df_plot["tf-idf"],
            color=COLORS[corp], edgecolor="white", linewidth=0.4)
    ax.set_title(f"{CORPUS_LABELS[corp]} ({corp})\n[{SEGMENT}]", fontsize=13)
    ax.set_xlabel("TF-IDF", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9)

fig.tight_layout()

out_path = FIGURES_DIR / f"esg_top_bigrams_{SEGMENT}.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {out_path.relative_to(ROOT)}")
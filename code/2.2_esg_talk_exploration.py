"""
ESG Talk Score — Distributions and Time Trends
===============================================

Plot the distribution of ESG-talk scores across firms and a time-trend
of average ESG talk from 2003 to 2024.

Input
-----
    data/processed/esg_talk.csv

Output
------
    output/figures/esg_talk_distributions.png
    output/figures/esg_talk_distributions_pres_vs_answers.png
    output/figures/esg_talk_time_trend.png

Usage
-----
    python code/2.2_esg_talk_exploration.py

Authors
-------
    Bastian Koch
"""

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR  = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── configuration ─────────────────────────────────────────────────────────────
SETS          = ("combined", "pres", "answers")
CORPORA       = ("e", "s", "g")
CORPUS_LABELS = {"e": "Environment", "s": "Social", "g": "Governance"}
SET_LABELS    = {"combined": "Combined", "pres": "Presentation", "answers": "Answers"}

_pal   = sns.color_palette("crest", 3)
COLORS = {"e": _pal[0], "s": _pal[1], "g": _pal[2]}

SCORE_COLS     = {s: [f"{c}_talk_{s}" for c in CORPORA] for s in SETS}
ALL_SCORE_COLS = [col for cols in SCORE_COLS.values() for col in cols]

# ── load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(PROC_DIR / "esg_talk.csv",
                 dtype={"permco": str, "permno": str, "gvkey": str})
df["year_call"] = df["year_call"].astype(int)
print(f"Rows: {len(df):,}   Columns: {len(df.columns)}")
print(f"Year range: {df['year_call'].min()} – {df['year_call'].max()}")


# ── 1. Score distributions (combined, top 1% clipped) ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, corp in zip(axes, CORPORA):
    col  = f"{corp}_talk_combined"
    vals = df[col]
    cap  = vals.quantile(0.99)
    ax.hist(vals.clip(upper=cap), bins=80, color=COLORS[corp],
            edgecolor="white", linewidth=0.3)
    ax.set_title(f"{CORPUS_LABELS[corp]}\n({col})", fontsize=12)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("# transcripts")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.text(0.97, 0.95,
            f"median={vals.median():.5f}\nmean={vals.mean():.5f}\nmax={vals.max():.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

fig.suptitle("ESG-Talk Score Distributions (combined, top 1% clipped)", fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "esg_talk_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → output/figures/esg_talk_distributions.png")


# ── 2. Presentation vs. Answers overlaid density histograms ──────────────────
def lighten(color, amount=0.55):
    c = mcolors.to_rgb(color)
    return tuple(1 - amount * (1 - x) for x in c)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, corp in zip(axes, CORPORA):
    pres_color = lighten(COLORS[corp], 0.55)
    ans_color  = COLORS[corp]

    for col_suf, color, label in [
        ("pres",    pres_color, "Presentation"),
        ("answers", ans_color,  "Answers"),
    ]:
        vals = df[f"{corp}_talk_{col_suf}"]
        cap  = vals.quantile(0.99)
        ax.hist(vals.clip(upper=cap), bins=80, color=color,
                alpha=0.65, edgecolor="none", label=label, density=True)

    ax.set_title(f"{CORPUS_LABELS[corp]}", fontsize=12)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(FIG_DIR / "esg_talk_distributions_pres_vs_answers.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → output/figures/esg_talk_distributions_pres_vs_answers.png")


# ── 3. Time trend: mean ESG-talk scores by year ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

for ax, seg in zip(axes, SETS):
    annual_seg = (df.groupby("year_call")[SCORE_COLS[seg]]
                    .mean()
                    .reset_index())
    for corp in CORPORA:
        col = f"{corp}_talk_{seg}"
        ax.plot(annual_seg["year_call"], annual_seg[col],
                marker="o", markersize=4, label=CORPUS_LABELS[corp],
                color=COLORS[corp])
    ax.set_title(f"Mean ESG-Talk Score by Year\n({SET_LABELS[seg]} segment)", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean cosine similarity")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(FIG_DIR / "esg_talk_time_trend.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → output/figures/esg_talk_time_trend.png")
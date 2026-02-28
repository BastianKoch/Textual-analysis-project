"""
score_transcripts_sbert.py
--------------------------
Compute Sentence-BERT ESG similarity scores for every processed earnings-call
transcript, using the pre-built reference vectors (E_vector.npy, S_vector.npy,
G_vector.npy).

Three text surfaces are scored per call:
  - Presentation   (_presentation.txt)
  - Q&A            (_answers.txt + _questions.txt concatenated)
  - Overall        (Presentation + Q&A concatenated)

Each surface is chunked (~1 200-1 800 chars). ALL chunks from ALL calls are
collected into a single flat list and encoded in one model.encode() call
(batch_size=64), then mean-pooled back to per-call unit vectors.
Cosine similarity against each reference vector (= dot product of two unit
vectors) gives the final scores.

Output columns (matches esg_scores.csv naming convention)
----------------------------------------------------------
  permco, permno, gvkey, comnam, filename, date_call,
  year_call, month_call, day_call,
  presentation_found, q&a_found,
  E_call,      S_call,      G_call         (whole call)
  pres_E_call, pres_S_call, pres_G_call    (presentation only)
  qa_E_call,   qa_S_call,   qa_G_call      (Q&A only)

Outputs
-------
  results/esg_call_similarity.parquet
  results/esg_call_similarity.csv

Expected runtime
----------------
  ~5 958 calls x 3 segments x ~12 chunks/segment = ~215 000 total chunks
  CPU (Apple M-series / modern x86):  ~20-40 min
  CUDA GPU:                           ~3-8 min

Usage
-----
    python src/score_transcripts_sbert.py
  or
    make score_sbert
"""

import csv
import io
import random
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---- Reproducibility ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

VECTORS_DIR  = DATA_DIR / "processed" / "esg_vectors"
SEGMENTS_DIR = DATA_DIR / "processed" / "Transcripts" / "Call_segments"
OVERVIEW_CSV = DATA_DIR / "processed" / "Transcripts" / "Overview_Calls.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PARQUET = RESULTS_DIR / "esg_call_similarity.parquet"
OUTPUT_CSV     = RESULTS_DIR / "esg_call_similarity.csv"

# ---- Model / chunking settings -----------------------------------------------
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE  = 64       # larger batch -> better CPU/GPU throughput
CHUNK_MIN   = 1_200    # characters, lower bound
CHUNK_MAX   = 1_800    # characters, upper bound


# ==============================================================================
# Text utilities
# ==============================================================================

def chunk_text(
    text: str,
    min_chars: int = CHUNK_MIN,
    max_chars: int = CHUNK_MAX,
) -> list[str]:
    """
    Split *text* into sentence-boundary-aware chunks (~min_chars-max_chars).

    No stopword removal, no stemming or lemmatisation; natural sentence
    structure is preserved. Returns [] for empty / whitespace-only input.
    """
    if not text or not text.strip():
        return []

    sentence_re = re.compile(r'(?<=[.!?])\s+')
    sentences: list[str] = []
    for raw in sentence_re.split(text.strip()):
        for line in raw.split('\n'):
            s = line.strip()
            if s:
                sentences.append(s)

    chunks: list[str] = []
    buffer = ""

    for sent in sentences:
        if not buffer:
            buffer = sent
            continue
        candidate = buffer + " " + sent
        if len(candidate) <= max_chars:
            buffer = candidate
            if len(buffer) >= min_chars:
                chunks.append(buffer.strip())
                buffer = ""
        else:
            if len(buffer) >= min_chars:
                chunks.append(buffer.strip())
                buffer = sent
            else:
                # Buffer still below minimum; absorb overshoot then hard-cut.
                combined = candidate
                while len(combined) > max_chars:
                    chunks.append(combined[:max_chars].strip())
                    combined = combined[max_chars:]
                buffer = combined

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if c]


def read_segment(path: Path) -> str:
    """Return file text, or '' if the file does not exist."""
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


# ==============================================================================
# Overview_Calls.csv parser
# ==============================================================================
# Mixed delimiter: first 9 columns inherited from the raw call list are
# comma-separated (with CSV quoting); all further columns appended by
# earnings_calls_processing.py use semicolons.

def _parse_overview_line(line: str, n_comma_cols: int = 9) -> list[str]:
    parts = line.rstrip("\n").split(";", maxsplit=n_comma_cols - 1)
    comma_fields = next(csv.reader(io.StringIO(parts[0])))
    return list(comma_fields) + parts[1:]


def load_overview(csv_path: Path) -> pd.DataFrame:
    with open(csv_path, encoding="utf-8") as fh:
        raw_lines = [l.rstrip("\n") for l in fh if l.strip()]

    header = _parse_overview_line(raw_lines[0])
    rows = []
    for line in raw_lines[1:]:
        fields = _parse_overview_line(line)
        while len(fields) < len(header):
            fields.append("")
        rows.append(fields[: len(header)])

    df = pd.DataFrame(rows, columns=header)
    # Derive filename stem used to locate segment files
    df["filename_stem"] = (
        df["filename"].str.strip('"').str.replace(".txt", "", regex=False)
    )
    return df


# ==============================================================================
# Aggregation helpers
# ==============================================================================

def mean_pool(embeddings: np.ndarray) -> Optional[np.ndarray]:
    """
    Mean-pool rows of *embeddings* and return an L2-normalised vector.
    Returns None for an empty (0-row) array.
    """
    if embeddings.shape[0] == 0:
        return None
    mv = embeddings.mean(axis=0)
    norm = np.linalg.norm(mv)
    if norm == 0 or np.isnan(norm):
        return None
    return mv / norm


def cosine_sim(a: Optional[np.ndarray], b: np.ndarray) -> Optional[float]:
    """Dot product of two unit vectors = cosine similarity."""
    if a is None:
        return None
    return float(np.dot(a, b))


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Model  : {MODEL_NAME}\n")

    # ---- Reference vectors ---------------------------------------------------
    print("Loading reference vectors ...")
    E_vec = np.load(VECTORS_DIR / "E_vector.npy")
    S_vec = np.load(VECTORS_DIR / "S_vector.npy")
    G_vec = np.load(VECTORS_DIR / "G_vector.npy")
    print(f"  E/S/G dim = {E_vec.shape[0]}\n")

    # ---- Load model ----------------------------------------------------------
    print("Loading SentenceTransformer ...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print()

    # ---- Load call metadata --------------------------------------------------
    print(f"Parsing {OVERVIEW_CSV.name} ...")
    meta = load_overview(OVERVIEW_CSV)
    total = len(meta)
    print(f"  {total:,} calls found.\n")

    # ==========================================================================
    # Phase 1 — read all segment files and split into chunks
    # ==========================================================================
    # Collect every chunk from every call/surface into ONE flat list so that
    # model.encode() can run a single vectorised pass over the full dataset,
    # avoiding repeated Python-side startup overhead.
    #
    # call_chunk_ranges[row_idx][surface] = (start, end)  -- slice into all_chunks

    print("Reading segment files and chunking ...")
    SURFACES = [("all", ""), ("pres", "pres_"), ("qa", "qa_")]

    all_chunks: list[str] = []
    call_chunk_ranges: list[dict[str, tuple[int, int]]] = []

    for _, row in tqdm(meta.iterrows(), total=total, desc="Chunking", unit="call"):
        stem = row["filename_stem"]

        pres_text  = read_segment(SEGMENTS_DIR / f"{stem}_presentation.txt")
        ans_text   = read_segment(SEGMENTS_DIR / f"{stem}_answers.txt")
        quest_text = read_segment(SEGMENTS_DIR / f"{stem}_questions.txt")
        qa_text    = "\n".join(t for t in [ans_text, quest_text] if t)
        full_text  = "\n".join(t for t in [pres_text, qa_text]   if t)

        row_ranges: dict[str, tuple[int, int]] = {}
        for surface, _prefix in SURFACES:
            text = {"all": full_text, "pres": pres_text, "qa": qa_text}[surface]
            chunks = chunk_text(text)
            start  = len(all_chunks)
            all_chunks.extend(chunks)
            row_ranges[surface] = (start, len(all_chunks))

        call_chunk_ranges.append(row_ranges)

    print(f"  Total chunks across all calls/surfaces: {len(all_chunks):,}\n")

    # ==========================================================================
    # Phase 2 — encode all chunks in a single model.encode() call
    # ==========================================================================
    print("Encoding all chunks (single pass) ...")
    all_embeddings: np.ndarray = model.encode(
        all_chunks,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,   # L2-normalise every chunk vector
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  Encoded {all_embeddings.shape[0]:,} chunks, dim={all_embeddings.shape[1]}\n")

    # ==========================================================================
    # Phase 3 — mean-pool per call/surface and compute cosine similarities
    # ==========================================================================
    print("Aggregating and scoring ...")

    META_COLS = [
        "permco", "permno", "gvkey", "comnam",
        "filename", "date_call", "year_call", "month_call", "day_call",
    ]
    META_COLS = [c for c in META_COLS if c in meta.columns]

    records: list[dict] = []

    for i, (_, row) in enumerate(meta.iterrows()):
        ranges = call_chunk_ranges[i]
        rec = {c: row[c] for c in META_COLS}
        rec["presentation_found"] = row.get("presentation_found", "")
        rec["q&a_found"]          = row.get("q&a_found", "")

        for surface, prefix in SURFACES:
            s, e = ranges[surface]
            vec = mean_pool(all_embeddings[s:e]) if e > s else None
            rec[f"{prefix}E_call"] = cosine_sim(vec, E_vec)
            rec[f"{prefix}S_call"] = cosine_sim(vec, S_vec)
            rec[f"{prefix}G_call"] = cosine_sim(vec, G_vec)

        records.append(rec)

    # ---- Assemble DataFrame --------------------------------------------------
    results = pd.DataFrame(records)

    for col in ("year_call", "month_call", "day_call",
                "presentation_found", "q&a_found"):
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")

    # ==========================================================================
    # Diagnostics
    # ==========================================================================
    score_cols = [
        "E_call",      "S_call",      "G_call",
        "pres_E_call", "pres_S_call", "pres_G_call",
        "qa_E_call",   "qa_S_call",   "qa_G_call",
    ]
    score_cols = [c for c in score_cols if c in results.columns]

    print("\n---- Summary statistics ----------------------------------------")
    desc = results[score_cols].describe(percentiles=[0.10, 0.50, 0.90])
    with pd.option_context(
        "display.float_format", "{:.6f}".format,
        "display.max_columns", 20,
        "display.width", 150,
    ):
        print(desc.loc[["mean", "std", "10%", "50%", "90%"]].to_string())

    print("\n---- Correlation matrix (whole-call scores) --------------------")
    with pd.option_context(
        "display.float_format", "{:.4f}".format,
        "display.width", 80,
    ):
        print(results[["E_call", "S_call", "G_call"]].corr().to_string())

    print("\n---- Missing / NaN counts ---------------------------------------")
    nan_counts = results[score_cols].isna().sum()
    nan_any = nan_counts[nan_counts > 0]
    if len(nan_any):
        for col, n in nan_any.items():
            print(f"  {col}: {n} NaN(s)")
    else:
        print("  None -- all calls scored successfully.")

    # ==========================================================================
    # Save
    # ==========================================================================
    print(f"\nSaving {len(results):,} rows ...")
    results.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    print(f"  Parquet -> {OUTPUT_PARQUET.relative_to(PROJECT_ROOT)}")
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"  CSV     -> {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()

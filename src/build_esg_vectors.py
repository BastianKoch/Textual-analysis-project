"""
build_esg_vectors.py
--------------------
Build one Sentence-BERT reference embedding vector per ESG corpus (E, S, G).

Pipeline
--------
1. Read cleaned .txt corpora from data/processed/Text corpus/{E,S,G}/
   - E: IPCC AR6 SYR Full Volume  +  IPCC AR6 SYR Longer Report (concatenated)
   - S: OECD Guidelines for Multinational Enterprises
   - G: OECD Corporate Governance Principles
2. Chunk each corpus into ~1 200–1 800-character windows on sentence boundaries.
3. Encode chunks with sentence-transformers (all-mpnet-base-v2),
   normalize_embeddings=True.
4. Mean-pool all chunk vectors → one fixed-length vector per corpus.
5. Validate: chunk counts, dimensionality, NaN check, pairwise cosine sims.
6. Save E_vector.npy, S_vector.npy, G_vector.npy to data/processed/esg_vectors/

Usage
-----
    python src/build_esg_vectors.py
    # or via Make:
    make build_esg_vectors
"""

import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(BASE, "data", "processed", "Text corpus")  # cleaned corpus
OUT_DIR = os.path.join(BASE, "data", "processed", "esg_vectors")
os.makedirs(OUT_DIR, exist_ok=True)

CORPUS_FILES = {
    # E: two files – concatenate them into one E corpus
    "E": [
        os.path.join(CORPUS_DIR, "E", "IPCC_AR6_SYR_FullVolume.txt"),
        os.path.join(CORPUS_DIR, "E", "ipcc_ar6_syr_longer_report.txt"),
    ],
    "S": [
        os.path.join(CORPUS_DIR, "S", "OECD_Guidelines.txt"),
    ],
    "G": [
        os.path.join(CORPUS_DIR, "G", "OECD_Corporate_Governance_Principles.txt"),
    ],
}

# ── Model settings ─────────────────────────────────────────────────────────────
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 32
CHUNK_MIN = 1200   # characters – lower bound for a chunk
CHUNK_MAX = 1800   # characters – upper bound; trim here if no sentence boundary


# ── Helper: read corpus ────────────────────────────────────────────────────────
def read_corpus(file_paths: list[str]) -> str:
    """Concatenate one or more .txt files into a single string."""
    parts = []
    for fp in file_paths:
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Corpus file not found: {fp}")
        with open(fp, encoding="utf-8") as fh:
            parts.append(fh.read())
    return "\n\n".join(parts)


# ── Helper: sentence-aware chunker ────────────────────────────────────────────
def chunk_text(text: str, min_chars: int = CHUNK_MIN, max_chars: int = CHUNK_MAX) -> list[str]:
    """
    Split *text* into chunks of approximately min_chars–max_chars characters.

    Strategy
    --------
    - Split text into sentences on `. `, `! `, `? `, newlines.
    - Accumulate sentences into a buffer until the buffer length reaches
      min_chars; then look for the next sentence boundary (up to max_chars).
    - Hard-break at max_chars if no sentence ending is found.
    - Stopwords are NOT removed; text is NOT stemmed or lemmatized.
    """
    # Tokenise into sentences (simple regex – good enough for normalised text)
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    sentences = sentence_endings.split(text.strip())
    # Also split on newlines that were not caught above
    split_sentences: list[str] = []
    for s in sentences:
        for line in s.split('\n'):
            stripped = line.strip()
            if stripped:
                split_sentences.append(stripped)

    chunks: list[str] = []
    buffer = ""

    for sent in split_sentences:
        if not buffer:
            buffer = sent
        else:
            candidate = buffer + " " + sent
            if len(candidate) <= max_chars:
                buffer = candidate
                if len(buffer) >= min_chars:
                    # Reached target size – save and reset
                    chunks.append(buffer.strip())
                    buffer = ""
            else:
                # Adding this sentence would exceed max_chars
                if len(buffer) >= min_chars:
                    chunks.append(buffer.strip())
                    buffer = sent
                else:
                    # Buffer still below min; accept the overshoot rather than
                    # producing a tiny chunk, then hard-break at max_chars.
                    combined = candidate
                    while len(combined) > max_chars:
                        chunks.append(combined[:max_chars].strip())
                        combined = combined[max_chars:]
                    buffer = combined

    if buffer.strip():
        # Flush remaining text (may be slightly below min_chars – keep it)
        chunks.append(buffer.strip())

    return [c for c in chunks if c]


# ── Helper: cosine similarity ──────────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors (= dot product)."""
    return float(np.dot(a, b))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model : {MODEL_NAME}\n")

    # Load model
    model = SentenceTransformer(MODEL_NAME, device=device)

    vectors: dict[str, np.ndarray] = {}
    chunk_counts: dict[str, int] = {}

    for label, file_paths in CORPUS_FILES.items():
        print(f"── Corpus {label} ──────────────────────")
        text = read_corpus(file_paths)
        chunks = chunk_text(text)
        chunk_counts[label] = len(chunks)
        print(f"  Files   : {[os.path.basename(p) for p in file_paths]}")
        print(f"  Length  : {len(text):,} chars")
        print(f"  Chunks  : {len(chunks)}")

        # Encode
        embeddings = model.encode(
            chunks,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,   # L2-normalise each chunk vector
            show_progress_bar=True,
            convert_to_numpy=True,
        )  # shape: (n_chunks, dim)

        # Mean-pool → one reference vector per corpus
        mean_vec = embeddings.mean(axis=0)

        # Re-normalise the mean vector so it is unit-length
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm

        vectors[label] = mean_vec
        print(f"  Embedding dim: {mean_vec.shape[0]}")
        print()

    # ── Validation ─────────────────────────────────────────────────────────────
    print("── Validation ──────────────────────────────")
    dim = next(iter(vectors.values())).shape[0]
    print(f"Embedding dimension : {dim}")
    for label, vec in vectors.items():
        nan_count = int(np.isnan(vec).sum())
        print(f"  {label} vector  shape={vec.shape}  NaNs={nan_count}  "
              f"chunks={chunk_counts[label]}")

    print()
    print("Pairwise cosine similarities (unit vectors → dot product):")
    labels = list(vectors.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            sim = cosine_sim(vectors[a], vectors[b])
            print(f"  cos({a}, {b}) = {sim:.6f}")

    # ── Save ────────────────────────────────────────────────────────────────────
    print()
    for label, vec in vectors.items():
        out_path = os.path.join(OUT_DIR, f"{label}_vector.npy")
        np.save(out_path, vec)
        print(f"Saved {label}_vector.npy  →  {os.path.relpath(out_path, BASE)}")

    print("\nDone.")


if __name__ == "__main__":
    main()

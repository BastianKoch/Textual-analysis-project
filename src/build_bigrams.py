"""
build_bigrams.py
----------------
For every normalized segment file in data/processed/Transcripts_nostop/
(presentation, answers, questions) build a list of consecutive word bigrams
and write them — one per line, space-separated — to data/interim/bigrams/.

Usage
-----
    python src/build_bigrams.py          # all segment types
    python src/build_bigrams.py pres     # only *_presentation.txt files

Output
------
    data/interim/bigrams/<same_filename>
    Each line:  word1 word2
"""

from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
INPUT_DIR    = DATA_DIR / "processed" / "Transcripts_nostop"
OUTPUT_DIR   = DATA_DIR / "interim"  / "bigrams"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Which segment types to process
# ---------------------------------------------------------------------------
_ALL_SUFFIXES = ("_presentation.txt", "_answers.txt", "_questions.txt")

arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
if arg == "pres":
    SUFFIXES = ("_presentation.txt",)
elif arg == "qa":
    SUFFIXES = ("_answers.txt", "_questions.txt")
else:
    SUFFIXES = _ALL_SUFFIXES


def build_bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    """Return every consecutive (token_i, token_i+1) pair."""
    return list(zip(tokens, tokens[1:]))


def process_file(src: Path, dst: Path) -> int:
    """Read a normalized token file, write bigrams. Returns bigram count."""
    text = src.read_text(encoding="utf-8").strip()
    if not text:
        dst.write_text("", encoding="utf-8")
        return 0
    tokens = text.split()
    bigrams = build_bigrams(tokens)
    dst.write_text("\n".join(f"{a} {b}" for a, b in bigrams), encoding="utf-8")
    return len(bigrams)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
files = sorted(
    f for f in INPUT_DIR.iterdir()
    if any(f.name.endswith(sfx) for sfx in SUFFIXES)
)

total = len(files)
print(f"Processing {total} files → {OUTPUT_DIR}")

for i, src in enumerate(files, 1):
    dst = OUTPUT_DIR / src.name
    n = process_file(src, dst)
    if i % 2000 == 0 or i == total:
        print(f"  {i}/{total}  {src.name}  ({n} bigrams)")

print("Done.")

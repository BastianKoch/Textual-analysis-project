"""
Build the ESG scoring dataset.

Reads the processed Overview_Calls.csv, loads _presentation.txt and
_answers.txt segments for every call, computes TF-IDF-weighted ESG topic
scores (Environmental / Social / Governance) for each segment type, and
writes a single merged CSV to data/processed/esg_scores.csv.

Output columns
--------------
  metadata    : permco, permno, gvkey, comnam, filename, date_call,
                year_call, month_call, day_call
  presentation: pres_esg_total, pres_esg_environmental,
                pres_esg_social, pres_esg_governance, pres_found
  q&a answers : ans_esg_total,  ans_esg_environmental,
                ans_esg_social,  ans_esg_governance,  ans_found

Usage
-----
    python src/build_esg_dataset.py
  or
    make esg_scores

@author: Bastian Koch
"""

import csv
import io
from pathlib import Path

import pandas as pd

from esg_scoring import compute_tfidf_esg_scores, load_esg_dict, load_segment_texts

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

OVERVIEW_CSV = DATA_DIR / "processed" / "Transcripts" / "Overview_Calls.csv"
SEGMENTS_DIR = DATA_DIR / "processed" / "Transcripts" / "Call_segments"
ESG_DICT_CSV = (
    DATA_DIR
    / "external"
    / "BaierBerningerKiesel_ESG-Wordlist_2020_July22.xlsx - ESG-Wordlist.csv"
)
OUTPUT_CSV = DATA_DIR / "processed" / "esg_scores.csv"


# =============================================================================
# Parse Overview_Calls.csv
# =============================================================================
# The file uses a mixed delimiter: the first 9 columns (carried over from the
# raw transcript list) are comma-separated with CSV quoting; the remaining
# columns appended by earnings_calls_processing.py are semicolon-separated.


def _parse_overview_line(line: str, n_comma_cols: int = 9) -> list[str]:
    """Split a mixed-delimiter line into a flat list of field values."""
    parts = line.rstrip("\n").split(";", maxsplit=n_comma_cols - 1)
    # parts[0] contains the first n_comma_cols fields as a comma-separated string
    comma_fields = next(csv.reader(io.StringIO(parts[0])))
    rest = parts[1:]
    return list(comma_fields) + rest


print("Loading Overview_Calls.csv ...", flush=True)
with open(OVERVIEW_CSV, encoding="utf-8") as f:
    raw_lines = [l.rstrip("\n") for l in f if l.strip()]

header_fields = _parse_overview_line(raw_lines[0])

rows = []
for line in raw_lines[1:]:
    fields = _parse_overview_line(line)
    # Pad short rows (calls with fewer managers than the maximum)
    while len(fields) < len(header_fields):
        fields.append("")
    rows.append(fields[: len(header_fields)])

overview_df = pd.DataFrame(rows, columns=header_fields)

# Keep only the core metadata columns
META_COLS = [
    "permco", "permno", "gvkey", "comnam",
    "filename", "date_call", "year_call", "month_call", "day_call",
    "presentation_found", "q&a_found", "number_analysts",
]
meta_cols_present = [c for c in META_COLS if c in overview_df.columns]
meta_df = overview_df[meta_cols_present].copy()

# Derive filename stem (strip .txt) used to locate segment files
meta_df["filename_stem"] = (
    meta_df["filename"].str.strip('"').str.replace(".txt", "", regex=False)
)

filenames = meta_df["filename_stem"].tolist()
total = len(filenames)
print(f"  {total} calls found in overview.", flush=True)

# =============================================================================
# Load ESG dictionary
# =============================================================================
print("Loading ESG dictionary ...", flush=True)
esg_dict = load_esg_dict(ESG_DICT_CSV)
print(f"  {len(esg_dict)} ESG words loaded across topics: "
      f"{sorted(set(esg_dict.values()))}", flush=True)

# =============================================================================
# Load segment texts
# =============================================================================
print("Loading presentation segments ...", flush=True)
pres_texts, pres_found = load_segment_texts(SEGMENTS_DIR, filenames, "presentation")
print(f"  {sum(pres_found)}/{total} presentation files found.", flush=True)

print("Loading Q&A answer segments ...", flush=True)
ans_texts, ans_found = load_segment_texts(SEGMENTS_DIR, filenames, "answers")
print(f"  {sum(ans_found)}/{total} answer files found.", flush=True)

# =============================================================================
# Compute TF-IDF ESG scores
# =============================================================================
print("Computing TF-IDF ESG scores for presentations ...", flush=True)
pres_scores = compute_tfidf_esg_scores(pres_texts, filenames, esg_dict)
pres_scores = pres_scores.add_prefix("pres_").rename(
    columns={"pres_filename": "filename_stem"}
)
pres_scores["pres_found"] = pres_found

print("Computing TF-IDF ESG scores for Q&A answers ...", flush=True)
ans_scores = compute_tfidf_esg_scores(ans_texts, filenames, esg_dict)
ans_scores = ans_scores.add_prefix("ans_").rename(
    columns={"ans_filename": "filename_stem"}
)
ans_scores["ans_found"] = ans_found

# =============================================================================
# Merge and write output
# =============================================================================
print("Merging and writing output ...", flush=True)
result = (
    meta_df
    .merge(pres_scores, on="filename_stem", how="left")
    .merge(ans_scores,  on="filename_stem", how="left")
    .drop(columns=["filename_stem"])
)

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_CSV, index=False)

print(f"\nDone. ESG scores written to: {OUTPUT_CSV}", flush=True)
print(f"Output shape: {result.shape[0]} rows × {result.shape[1]} columns", flush=True)

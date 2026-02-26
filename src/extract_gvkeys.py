"""
Extract firm identifiers from Overview_Calls.csv into separate CSV files.

Outputs
-------
data/processed/gvkeys.csv   — unique gvkeys  (columns: gvkey, comnam)
data/processed/permnos.csv  — unique permnos (columns: permno, permco, comnam)

Usage
-----
    python src/extract_gvkeys.py
  or
    make gvkeys

@author: Bastian Koch
"""

import csv
import io
# csv used only for parsing; no writer needed for plain-text output
from pathlib import Path

# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
OVERVIEW_CSV  = PROJECT_ROOT / "data" / "processed" / "Transcripts" / "Overview_Calls.csv"
GVKEY_TXT     = PROJECT_ROOT / "data" / "interim" / "gvkeys.txt"
PERMNO_TXT    = PROJECT_ROOT / "data" / "interim" / "permnos.txt"


# =============================================================================
# Parse Overview_Calls.csv
# The file has a mixed delimiter: the first 9 fields are comma-separated,
# the remainder are semicolon-separated.
# =============================================================================
def parse_overview_line(line: str) -> list[str]:
    parts = line.rstrip("\n").split(";", maxsplit=8)
    comma_fields = next(csv.reader(io.StringIO(parts[0])))
    return list(comma_fields)


print("Reading Overview_Calls.csv ...", flush=True)
with open(OVERVIEW_CSV, encoding="utf-8") as f:
    raw_lines = [l.rstrip("\n") for l in f if l.strip()]

header = parse_overview_line(raw_lines[0])
# Expected: permco, permno, gvkey, comnam, filename, date_call, year_call, month_call, day_call
permco_idx = header.index("permco")
permno_idx = header.index("permno")
gvkey_idx  = header.index("gvkey")
comnam_idx = header.index("comnam")

gvkeys  = set()
permnos = set()

for line in raw_lines[1:]:
    fields = parse_overview_line(line)
    if len(fields) <= max(permco_idx, permno_idx, gvkey_idx, comnam_idx):
        continue

    gvkey  = fields[gvkey_idx].strip('"').strip()
    permno = fields[permno_idx].strip('"').strip()

    if gvkey:
        gvkeys.add(gvkey)
    if permno:
        permnos.add(permno)

# =============================================================================
# Write outputs
# =============================================================================
GVKEY_TXT.parent.mkdir(parents=True, exist_ok=True)

with open(GVKEY_TXT, "w", encoding="utf-8") as f:
    for gvkey in sorted(gvkeys):
        f.write(gvkey + "\n")

with open(PERMNO_TXT, "w", encoding="utf-8") as f:
    for permno in sorted(permnos, key=lambda x: x):
        f.write(permno + "\n")

print(f"Done.", flush=True)
print(f"  {len(gvkeys):,}  unique gvkeys  → {GVKEY_TXT}", flush=True)
print(f"  {len(permnos):,} unique permnos → {PERMNO_TXT}", flush=True)

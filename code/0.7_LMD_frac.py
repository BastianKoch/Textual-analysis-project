"""
LMD Fractions from Earnings Call Transcripts
=============================================

Calculate Loughran-McDonald (2011) word-list fractions (NEGFRAC, LITFRAC,
UNCFRAC) and log word count (LOGWORD) for each processed earnings call
transcript segment.

Input
-----
    data/raw/list_earnings_calls_group_project_upload.csv
    data/processed/Transcripts/{year}/*.txt
    data/external/LMD/LMD_Neg.txt
    data/external/LMD/LMD_Litigious.txt
    data/external/LMD/LMD_Uncertainty.txt

Output
------
    data/processed/LMD_Frac.csv

Usage
-----
    python code/0.7_LMD_frac.py

Authors
-------
    Aryan
"""

import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


ROOT            = Path(__file__).resolve().parents[1]
OVERVIEW_FILE   = ROOT / "data" / "raw" / "list_earnings_calls_group_project_upload.csv"
SEGMENTS_FOLDER = ROOT / "data" / "processed" / "Transcripts"
LM_NEG_WORDS    = ROOT / "data" / "external" / "LMD" / "LMD_Neg.txt"
LM_LIT_WORDS    = ROOT / "data" / "external" / "LMD" / "LMD_Litigious.txt"
LM_UNC_WORDS    = ROOT / "data" / "external" / "LMD" / "LMD_Uncertainty.txt"
OUTPUT_FILE     = ROOT / "data" / "processed" / "LMD_Frac.csv"
# ─────────────────────────────────────────────

# ── 1. Load LM word lists ────────────────────────────────────────────
print("Loading LM word lists...")
with open(LM_NEG_WORDS, 'r', encoding='utf-8') as f:
    lm_neg = set(line.strip().upper() for line in f if line.strip())
print(f"  {len(lm_neg):,} negative words loaded")

with open(LM_LIT_WORDS, 'r', encoding='utf-8') as f:
    lm_lit = set(line.strip().upper() for line in f if line.strip())
print(f"  {len(lm_lit):,} litigious words loaded")

with open(LM_UNC_WORDS, 'r', encoding='utf-8') as f:
    lm_unc = set(line.strip().upper() for line in f if line.strip())
print(f"  {len(lm_unc):,} uncertainty words loaded")

# ── 2. Load Overview_Calls to get call metadata ───────────────────────────────
print("\nLoading Overview_Calls...")
overview = pd.read_csv(
    OVERVIEW_FILE,
    sep=',',
    header=0,
    names=['permco', 'permno', 'gvkey', 'comnam', 'filename', 'date_call', 'year_call', 'month_call'],
    usecols=[0, 1, 2, 3, 4, 5, 6, 7],   # only read first 8 columns
    engine='python',
    encoding='utf-8',
    on_bad_lines='skip',
    skiprows=1,
    index_col=False
)
overview.columns = overview.columns.str.strip().str.lower().str.replace(' ', '_')
print(f"  {len(overview):,} calls in overview")
print(f"  Columns: {overview.columns.tolist()}")


# ── 3. Helper: extract text from a segment excel file ────────────────────────
def read_segment_text(filepath):
    """Read a txt segment file and return its text."""
    if not os.path.exists(filepath):
        return ""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"  Warning reading {filepath}: {e}")
        return ""
    

# ── 4. Helper: calculate NEGFRAC and LOGWORD from raw text ───────────────────
def tokenize(text):
    """Uppercase words only, strip punctuation."""
    return re.findall(r'\b[A-Za-z]+\b', text.upper())

def calc_metrics(text):
    words = tokenize(text)
    total = len(words)
    if total == 0:
        return np.nan, np.nan, np.nan, np.nan
    negfrac = sum(1 for w in words if w in lm_neg) / total
    litfrac = sum(1 for w in words if w in lm_lit) / total
    uncfrac = sum(1 for w in words if w in lm_unc) / total
    logword = np.log(total)
    return negfrac, litfrac, uncfrac, logword

# ── 5. Loop over calls and compute metrics ────────────────────────────────────
print("\nProcessing call segments...")

results = []
segments_path = Path(SEGMENTS_FOLDER)

id_col = 'filename'
print(f"  Using '{id_col}' as call identifier")

for i, row in overview.iterrows():
    call_id = str(row[id_col]).strip().replace('.txt', '')  # e.g. "1665733"
    
    pres_file = segments_path / f"{call_id}_presentation.txt"
    ans_file  = segments_path / f"{call_id}_answers.txt"
    ques_file = segments_path / f"{call_id}_questions.txt"

    pres_text = read_segment_text(str(pres_file))
    ans_text  = read_segment_text(str(ans_file))

    # Full call = presentation + answers (standard in literature)
    full_text = pres_text + ' ' + ans_text

    if full_text.strip() == "":
        print(f"  WARNING: no text found for {call_id}")

    negfrac_full, litfrac_full, uncfrac_full, logword_full = calc_metrics(full_text)
    negfrac_pres, litfrac_pres, uncfrac_pres, logword_pres = calc_metrics(pres_text)
    negfrac_ans,  litfrac_ans,  uncfrac_ans,  logword_ans  = calc_metrics(ans_text)

    result_row = row.to_dict()
    result_row.update({
    'NEGFRAC':      negfrac_full,
    'LITFRAC':      litfrac_full,
    'UNCFRAC':      uncfrac_full,
    'LOGWORD':      logword_full,
    'NEGFRAC_pres': negfrac_pres,
    'LITFRAC_pres': litfrac_pres,
    'UNCFRAC_pres': uncfrac_pres,
    'LOGWORD_pres': logword_pres,
    'NEGFRAC_ans':  negfrac_ans,
    'LITFRAC_ans':  litfrac_ans,
    'UNCFRAC_ans':  uncfrac_ans,
    'LOGWORD_ans':  logword_ans,
    })
    results.append(result_row)

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(overview)} calls...")
        


# ── 6. Save output ────────────────────────────────────────────────────────────
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nDone! Output saved to: {OUTPUT_FILE}")
print(f"  Total calls processed: {len(out_df):,}")
print(f"  Missing NEGFRAC: {out_df['NEGFRAC'].isna().sum():,}")
print(f"\nSample output:")
print(out_df[['NEGFRAC', 'LITFRAC', 'UNCFRAC' , 'LOGWORD']].describe().round(4))

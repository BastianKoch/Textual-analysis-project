"""
Extract and clean text from the IPCC AR6 Synthesis Report Full Volume PDF.

Input:  data/external/Text corpus/E/IPCC_AR6_SYR_FullVolume.pdf
Output: data/interim/Text corpus/E/IPCC_AR6_SYR_FullVolume.txt

Usage
-----
    python src/ipcc_pdf_extractor.py
"""

import re
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "external" / "Text corpus" / "E" / "IPCC_AR6_SYR_FullVolume.pdf"
OUT_PATH = ROOT / "data" / "interim" / "Text corpus" / "E" / "IPCC_AR6_SYR_FullVolume.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Extract text page by page
# ---------------------------------------------------------------------------
print(f"Opening {PDF_PATH.name} ({PDF_PATH.stat().st_size // 1024} KB) ...", flush=True)
doc = fitz.open(PDF_PATH)
print(f"  {doc.page_count} pages", flush=True)

raw = "\n".join(page.get_text() for page in doc)

# ---------------------------------------------------------------------------
# Clean up
# ---------------------------------------------------------------------------
# Normalise whitespace characters
raw = raw.replace("\xa0", " ").replace("\xad", "")

# Strip leading/trailing whitespace on each line
lines = [line.strip() for line in raw.splitlines()]
raw = "\n".join(lines)

# Remove standalone page numbers (arabic) and roman numerals
raw = re.sub(r"^\d{1,3}$", "", raw, flags=re.MULTILINE)
raw = re.sub(r"^[ivxlcIVXLC]{1,6}$", "", raw, flags=re.MULTILINE)

# Remove standalone decimal numbers (chart axis labels: "0.5", "1.5", "2.0")
raw = re.sub(r"^-?\d+\.\d+$", "", raw, flags=re.MULTILINE)

# Remove repeating running headers / section labels that appear at the top of
# each page.  Pattern: a section title is printed once (or twice in a row)
# followed by a "Section N" or "Annexes" label.
# Step 1 – remove duplicate consecutive non-blank lines (the doubled titles)
def remove_consecutive_duplicates(text):
    lines = text.splitlines()
    out = []
    prev = None
    for line in lines:
        if line.strip() and line == prev:
            continue          # skip exact duplicate of previous non-blank line
        out.append(line)
        if line.strip():
            prev = line
        else:
            prev = None       # reset on blank line
    return "\n".join(out)

raw = remove_consecutive_duplicates(raw)

# Step 2 – remove known section-label lines that appear as headers
SECTION_HEADERS = [
    r"^Section \d+$",
    r"^Annexes$",
    r"^Summary for Policymakers$",
    r"^Current Status and Trends$",
    r"^Long-Term Climate and Development Futures$",
    r"^Near-Term Responses in a Changing Climate$",
    r"^Synthesis and Integration$",
    r"^Expert Reviewers AR6 SYR$",
    r"^CLIMATE CHANGE 2023$",
    r"^Synthesis Report$",
    r"^A Report of the Intergovernmental Panel on Climate Change$",
    # TOC section-category labels
    r"^Contents$",
    r"^Front matter$",
    r"^SPM$",
    r"^Sections$",
]
for pattern in SECTION_HEADERS:
    raw = re.sub(pattern, "", raw, flags=re.MULTILINE)

# Remove IPCC cross-reference citation blocks: {2.1.1}, {WGI SPM A.1 ...}
raw = re.sub(r"\{[^}]{1,80}\}", "", raw)

# Strip front matter: remove everything before the first "Foreword and Preface"
foreword_match = re.search(r"^Foreword and Preface$", raw, re.MULTILINE)
if foreword_match:
    raw = raw[foreword_match.start():]

# Strip back matter: remove everything from Annex I (Glossary) onwards
# The glossary section starts with "Annex I\nGlossary" as standalone lines
annex_match = re.search(r"^Annex I\nGlossary$", raw, re.MULTILINE)
if annex_match:
    raw = raw[:annex_match.start()]

# Remove table of contents: lines with 5+ consecutive dashes (TOC entries)
# e.g. "Foreword  ----... v"  or  "Section 2 ----... 41"
raw = re.sub(r"^.{2,100}-{5,}.*$", "", raw, flags=re.MULTILINE)

# ---------------------------------------------------------------------------
# Remove figure/table data fragments (axis labels, legends, chart values)
# ---------------------------------------------------------------------------
# 1. Lines consisting only of bullet/dot rating symbols (•••, ••, etc.)
raw = re.sub(r"^[•·\u25e6●\u25cb◆◇■□▪▫\s]+$", "", raw, flags=re.MULTILINE)

# 2. Bullets followed by short figure labels: "•• damage", "••• Wildfire"
raw = re.sub(r"^[•·●◆▪\s]{2,}\s*\w[\w\s]{0,25}$", "", raw, flags=re.MULTILINE)

# 3. Standalone °C or °C with dash/en-dash (temperature axis tick)
raw = re.sub(r"^[°C\s\u2013\-]+$", "", raw, flags=re.MULTILINE)

# 4. Standalone risk-rating letters used in RFC figures: bare "r" or "R"
raw = re.sub(r"^[rR]$", "", raw, flags=re.MULTILINE)

# 5. Percentage axis tick labels: +35%, -20%, +3, -10 etc. (signed number, ≤ 5 chars)
raw = re.sub(r"^[+\-\u2013]\d{1,3}(\.\d+)?%?$", "", raw, flags=re.MULTILINE)

# 6. Chart legend labels for RFC risk levels
raw = re.sub(
    r"^(very )?(low|high)( (low|high))?$",
    "", raw, flags=re.MULTILINE | re.IGNORECASE
)
raw = re.sub(r"^intermediate (low|high)$", "", raw, flags=re.MULTILINE | re.IGNORECASE)

# 7. Year-sequence lines (axis ticks like "2000 2015 2050 2100")
raw = re.sub(r"^\d{4}(\s+\d{4})+$", "", raw, flags=re.MULTILINE)

# 8. Lines of only "0 days", "365 days", "0%", "100%" (infographic duration/pct labels)
raw = re.sub(r"^\d+\s*(days?|%)$", "", raw, flags=re.MULTILINE | re.IGNORECASE)

# 9. Specific known figure/chart labels
raw = re.sub(r"^AR5 AR6$", "", raw, flags=re.MULTILINE)           # RFC comparison label
raw = re.sub(r"^intermediate$", "", raw, flags=re.MULTILINE)       # RFC legend fragment

# 10. Remove blocks of figure axis text: runs of 5+ consecutive non-blank lines
#     that are each ≤ 25 characters, contain no sentence-ending punctuation, and
#     don't start with a digit. These are almost always chart axis label lists.
def remove_axis_label_blocks(text, min_run=5, max_len=25):
    lines = text.splitlines()

    def is_axis_candidate(line):
        s = line.strip()
        if not s or len(s) > max_len:
            return False
        if re.search(r'[.!?:;]', s):   # has punctuation → real sentence fragment
            return False
        if re.match(r'^\d', s):        # starts with digit → stat / year
            return False
        return True

    to_remove = set()
    n = len(lines)
    i = 0
    while i < n:
        if is_axis_candidate(lines[i]):
            run = []
            j = i
            # Collect consecutive axis-candidate non-blank lines
            while j < n and lines[j].strip() and is_axis_candidate(lines[j]):
                run.append(j)
                j += 1
            if len(run) >= min_run:
                for idx in run:
                    to_remove.add(idx)
            i = j if j > i else i + 1
        else:
            i += 1

    return "\n".join("" if idx in to_remove else l for idx, l in enumerate(lines))

raw = remove_axis_label_blocks(raw)

# ---- Chemical formulas: join subscripts BEFORE footnote removal ----
# CO\n2, CH\n4, N\n2\nO, NO\n2, H\n2\nO, SO\n2 etc.
raw = re.sub(r"\b(CO|CH|SO|NO|N)\n(\d)", r"\1\2", raw)
raw = re.sub(r"\b(N2O|CO2|CH4|SO2|NO2|H2O|GtCO2|MtCO2|tCO2)\n(-eq|-FFI|-LULUCF)",
             r"\1\2", raw)

# Join unit suffixes split across lines: GtCO2\n-eq yr\n–1
raw = re.sub(r"\n(-eq|-FFI|-LULUCF)(?=\b)", r"\1", raw)
raw = re.sub(r"\byr\n([–\-]\s*1\b)", r"yr\1", raw)

# Remove superscript-style footnote reference numbers embedded in running text
# (lone digits on their own line that are clearly footnote markers)
raw = re.sub(r"(?<=\S)\n(\d{1,2})\n(?=\S)", r"\n", raw)  # footnote between words

# Join hyphenated line breaks: "sustain-\nable" → "sustainable"
raw = re.sub(r"-\n(\w)", r"\1", raw)

# Join en-dash date ranges split across lines: 1850\n–\n1900 → 1850–1900
raw = re.sub(r"(\d)\n([–\-])\n(\d)", r"\1\2\3", raw)

# Join orphaned punctuation onto previous line: comma/period alone on a line
raw = re.sub(r"\n([,;])", r"\1", raw)

# Join mid-sentence broken lines (line ends without stop, next starts lowercase)
def join_broken_lines(text):
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            out.append(line)
            i += 1
            continue
        # Look at the next non-blank line
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            next_line = lines[j].strip()
            ends_without_stop = not re.search(r"[.!?:]\s*$", line)
            starts_lowercase = next_line and next_line[0].islower()
            gap = j - i - 1  # number of blank lines in between
            if ends_without_stop and starts_lowercase and gap <= 1:
                out.append(line + " " + next_line)
                i = j + 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out)

raw = join_broken_lines(raw)

# Collapse 3+ consecutive blank lines to a single blank line
raw = re.sub(r"\n{3,}", "\n\n", raw)

text = raw.strip()

# ---------------------------------------------------------------------------
# Normalise text for NLP: remove numbers and special characters
# ---------------------------------------------------------------------------
def normalize_for_nlp(text):
    lines = text.splitlines()
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue

        # 1. Remove all digit characters
        line = re.sub(r"\d", "", line)

        # 2. Remove scientific / typographic symbols
        #    °  %  ±  ≤  ≥  ×  ÷  ≈  ≠  ∞  →  ←  ↑  ↓  √  ∑  •  ·
        #    em-dash  en-dash  non-breaking space ligatures etc.
        line = re.sub(r"[°%±≤≥×÷≈≠∞→←↑↓√∑•·–—\u00b0\u2013\u2014\u2022\u00b7]", " ", line)

        # 3. Remove any remaining non-ASCII characters
        line = line.encode("ascii", errors="ignore").decode("ascii")

        # 4. Remove characters that aren't letters, spaces, hyphens (within
        #    words), apostrophes (contractions), or basic sentence punctuation
        line = re.sub(r"[^a-zA-Z\s\-\'.,:;!?()]", " ", line)

        # 5. Clean up orphaned hyphens / apostrophes (not flanked by letters)
        line = re.sub(r"(?<![a-zA-Z])[-']|[-'](?![a-zA-Z])", " ", line)

        # 6. Remove empty parentheses / brackets left after stripping
        line = re.sub(r"\(\s*\)", " ", line)
        line = re.sub(r",\s*,", ",", line)   # double commas

        # 7. Collapse multiple spaces
        line = re.sub(r" {2,}", " ", line).strip()

        # 8. Drop lines that are now too short to be meaningful (< 3 words)
        if len(line.split()) < 3:
            out.append("")
            continue

        out.append(line)

    # Collapse extra blank lines that remain after stripping
    joined = "\n".join(out)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()

text = normalize_for_nlp(text)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_PATH.write_text(text, encoding="utf-8")
print(f"Saved {len(text):,} characters ({text.count(chr(10)):,} lines) to {OUT_PATH}")

"""
Extract and clean text from the G20/OECD Principles of Corporate Governance PDF.

Input:  data/external/Text corpus/G/OECD_Corporate Governance Principles.pdf
Output: data/interim/Text corpus/G/OECD_Corporate_Governance_Principles.txt

Usage
-----
    python src/governance_pdf_extractor.py
"""

import re
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "external" / "Text corpus" / "G" / "OECD_Corporate Governance Principles.pdf"
OUT_PATH = ROOT / "data" / "interim" / "Text corpus" / "G" / "OECD_Corporate_Governance_Principles.txt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Extract text page by page (skip back cover – last page is ISBN/barcode only)
# ---------------------------------------------------------------------------
print(f"Opening {PDF_PATH.name} ({PDF_PATH.stat().st_size // 1024} KB) ...", flush=True)
doc = fitz.open(PDF_PATH)
print(f"  {doc.page_count} pages", flush=True)

raw = "\n".join(page.get_text() for page in doc)

# ---------------------------------------------------------------------------
# Clean up
# ---------------------------------------------------------------------------
# Replace non-breaking spaces, soft hyphens, and the private-use bullet \uf07c
raw = raw.replace("\xa0", " ").replace("\xad", "").replace("\uf07c", "")

# Strip leading/trailing whitespace on each line
lines = [line.strip() for line in raw.splitlines()]
raw = "\n".join(lines)

# Remove standalone page numbers (arabic) and roman numerals
raw = re.sub(r"^\d{1,3}$", "", raw, flags=re.MULTILINE)
raw = re.sub(r"^[ivxlcIVXLC]{1,6}$", "", raw, flags=re.MULTILINE)

# Remove running headers/footers that repeat on every page
for pattern in [
    r"^G20/OECD PRINCIPLES OF CORPORATE GOVERNANCE.*$",
    r"^G20/OECD Principles of Corporate Governance.*$",
    r"^G20/OECD Principles\s+of\s+Corporate Governance.*$",
    r"^PRINT ISBN.*$",
    r"^PDF ISBN.*$",
    r"^9HSTCQE\*.*$",         # back-cover barcode/ISBN string
    r"^\[OECD/LEGAL/\d+\].*$",
]:
    raw = re.sub(pattern, "", raw, flags=re.MULTILINE | re.IGNORECASE)

# Join hyphenated line breaks: "sustain-\nable" → "sustainable"
raw = re.sub(r"-\n(\w)", r"\1", raw)

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
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            next_line = lines[j].strip()
            ends_without_stop = not re.search(r"[.!?:]\s*$", line)
            starts_lowercase = next_line and next_line[0].islower()
            gap = j - i - 1
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
        line = re.sub(r"\d", "", line)
        line = re.sub(r"[\u00b0%\u00b1\u2264\u2265\u00d7\u00f7\u2248\u2260\u221e"
                      r"\u2192\u2190\u2191\u2193\u221a\u2211\u2022\u00b7\u2013\u2014]", " ", line)
        line = line.encode("ascii", errors="ignore").decode("ascii")
        line = re.sub(r"[^a-zA-Z\s\-\'.,:;!?()]", " ", line)
        line = re.sub(r"(?<![a-zA-Z])[-']|[-'](?![a-zA-Z])", " ", line)
        line = re.sub(r"\(\s*\)", " ", line)
        line = re.sub(r" {2,}", " ", line).strip()
        if len(line.split()) < 3:
            out.append("")
            continue
        out.append(line)
    joined = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()

text = normalize_for_nlp(text)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
OUT_PATH.write_text(text, encoding="utf-8")
print(f"Saved {len(text):,} characters ({text.count(chr(10)):,} lines) to {OUT_PATH}")

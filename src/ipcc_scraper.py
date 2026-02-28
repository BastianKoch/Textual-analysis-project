"""
Scrape the IPCC AR6 Synthesis Report (Longer Report) and save cleaned plain text.

Output: data/processed/Text corpus/E/ipcc_ar6_syr_longer_report.txt

Usage
-----
    python src/ipcc_scraper.py
"""

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = "https://www.ipcc.ch/report/ar6/syr/longer-report/"

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
print(f"Fetching {URL} ...", flush=True)
html = requests.get(URL, timeout=30).text
soup = BeautifulSoup(html, "html.parser")

# ---------------------------------------------------------------------------
# Remove non-content elements before extracting text
# ---------------------------------------------------------------------------
for tag in soup.find_all(["nav", "header", "footer", "script", "style",
                           "noscript", "button", "svg", "figure"]):
    tag.decompose()

# Remove social-share / UI-only elements by class name patterns
for tag in soup.find_all(True, class_=re.compile(
        r"share|social|menu|nav|sidebar|cookie|banner|modal|overlay|figure", re.I)):
    tag.decompose()

# ---------------------------------------------------------------------------
# Extract text from main content area
# ---------------------------------------------------------------------------
main = soup.find("article") or soup.find("main") or soup.find("body")
raw = main.get_text(separator="\n")

# ---------------------------------------------------------------------------
# Clean up
# ---------------------------------------------------------------------------
# Replace non-breaking spaces and zero-width spaces
raw = raw.replace("\xa0", " ").replace("\u200b", "")

# Remove leftover UI artefacts
for pattern in [
    r"View details",
    r"Open figure",
    r"Expand section",
    r"Collapse section",
    r"Copy link",
    r"Share via email",
    r"Share on \w+",
    r"Copy\s*doi",
    r"Read online",
    r"Download",
]:
    raw = re.sub(pattern, "", raw, flags=re.IGNORECASE)

# Strip leading/trailing whitespace on each line first
lines = [line.strip() for line in raw.splitlines()]
raw = "\n".join(lines)

# Remove IPCC cross-reference citation blocks: { WGI SPM ... }
raw = re.sub(r"\{[^}]{0,500}\}", "", raw, flags=re.DOTALL)

# Join chemical formula subscripts onto the previous line BEFORE footnote removal
# so that CO\n2 → CO2 before the lone "2" is accidentally deleted
raw = re.sub(r"([A-Z]{1,3})\n(\d)\b", r"\1\2", raw)

# Join unit suffixes broken onto their own line:
# GtCO2\n-eq → GtCO2-eq,  yr\n–1 → yr–1,  CO2\n-FFI → CO2-FFI
raw = re.sub(r"(\w)\n(-eq|-FFI|-LULUCF|-eq yr)", r"\1\2", raw)
raw = re.sub(r"(yr)\n([–-]\d)", r"\1\2", raw)
raw = re.sub(r"(\d)\n([–-]\d)", r"\1\2", raw)

# Remove standalone footnote numbers (a line containing only digits)
raw = re.sub(r"^\d{1,3}$", "", raw, flags=re.MULTILINE)

# Join confidence labels broken across lines: (\n high confidence \n) → (high confidence)
raw = re.sub(r"\(\s*\n\s*((?:very |extremely |virtually )?high confidence|medium confidence|low confidence|likely|very likely|unlikely|virtually certain|extremely likely)\s*\n\s*\)", r"(\1)", raw)

# Join dashes/en-dashes on their own line back to surrounding text: 1850\n–\n1900 → 1850–1900
raw = re.sub(r"(\w)\s*\n\s*([-–])\s*\n\s*(\w)", r"\1\2\3", raw)

# Join orphan leading punctuation (. , ) onto previous line
raw = re.sub(r"\n([.,;:\)])", r"\1\n", raw)

# Join inline IPCC calibrated language terms on their own line back to surrounding text
# e.g. "It is\nlikely\nthat" → "It is likely that"
_calibrated = (r"very high confidence|high confidence|medium confidence|low confidence"
               r"|virtually certain|extremely likely|very likely|likely|unlikely"
               r"|very unlikely|exceptionally unlikely")
raw = re.sub(rf"\n({_calibrated})\n", r" \1 ", raw, flags=re.IGNORECASE)

# Join broken quoted section titles: '\nFoo Bar\n' → 'Foo Bar'
raw = re.sub(r"'\s*\n\s*(.+?)\s*\n\s*'", r"'\1'", raw)

# Remove the front-matter header (everything before "1. Introduction")
intro_match = re.search(r"^1\.\s+Introduction", raw, flags=re.MULTILINE)
if intro_match:
    raw = raw[intro_match.start():]

# Collapse 3+ consecutive blank lines to a single blank line
raw = re.sub(r"\n{3,}", "\n\n", raw)

# Final pass: join mid-sentence blank lines — if a line doesn't end in terminal
# punctuation and the next non-empty line starts lowercase, join them
def join_broken_sentences(text):
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # If line is blank, just add it
        if not line.strip():
            out.append(line)
            i += 1
            continue
        # Peek ahead past any blank lines
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            next_line = lines[j].strip()
            ends_without_stop = line and not re.search(r'[.!?:]\s*$', line)
            starts_lowercase = next_line and next_line[0].islower()
            # If gap has exactly (j - i - 1) blank lines and it looks like mid-sentence
            if ends_without_stop and starts_lowercase and (j - i - 1) <= 1:
                # Merge: output line + space + next, skip the blanks
                out.append(line + " " + next_line)
                i = j + 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out)

text = join_broken_sentences(raw).strip()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = (
    Path(__file__).resolve().parents[1]
    / "data" / "processed" / "Text corpus" / "E" / "ipcc_ar6_syr_longer_report.txt"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(text, encoding="utf-8")

print(f"Saved {len(text):,} characters ({text.count(chr(10)):,} lines) to {out_path}")

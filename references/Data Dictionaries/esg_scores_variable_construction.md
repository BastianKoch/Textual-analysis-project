# ESG Score Variables — Construction and Interpretation

**Dataset:** `data/processed/esg_scores.csv`  
**Produced by:** `src/build_esg_dataset.py` (run via `make esg_scores`)  
**Dictionary source:** Baier, Berninger & Kiesel (2020), *"ESG-Wordlist"*, available at the Journal of Asset Management.

---

## Overview

Each row in `esg_scores.csv` corresponds to one earnings conference call. For each call, we measure how much ESG-related language managers use in two distinct segments — the **prepared presentation** and the **Q&A answers**. Scores are constructed using TF-IDF weighting over a curated ESG dictionary.

---

## Source Dictionary

The Baier/Berninger/Kiesel (2020) ESG word list contains **491 words** categorised into three topics:

| Topic | Column value | Example words |
|---|---|---|
| Environmental | `Environmental` | emissions, biodiversity, carbon, renewable, pollution |
| Social | `Social` | diversity, safety, employees, community, labor |
| Governance | `Governance` | board, audit, transparency, shareholders, compliance |

File location: `data/external/BaierBerningerKiesel_ESG-Wordlist_2020_July22.xlsx - ESG-Wordlist.csv`

---

## Text Segments

Each transcript is split into two segments by `src/earnings_calls_processing.py`:

| Segment | Suffix | Description |
|---|---|---|
| Presentation | `_presentation.txt` | Prepared remarks by management before Q&A |
| Q&A Answers | `_answers.txt` | Responses given by management during Q&A |

Q&A questions (from analysts) are excluded — we want to capture *management's* language only.

---

## TF-IDF Scoring Procedure

Scores are computed separately for presentations and Q&A answers using scikit-learn's `TfidfVectorizer` with the following settings:

- **Vocabulary**: restricted to the 491 ESG dictionary words (unigrams only)
- **Corpus**: all available segment texts are used together to fit the IDF weights
- **Normalisation**: L2 norm per document (default), so document length is partially controlled for

For each document $d$ and each ESG word $w$:

$$\text{tfidf}(w, d) = \text{tf}(w, d) \times \log\!\left(\frac{1 + N}{1 + \text{df}(w)}\right) + 1$$

where $N$ is the total number of documents and $\text{df}(w)$ is the number of documents containing $w$.

Words that appear in nearly every call (e.g., "board") receive a low IDF weight. Words that are rare but ESG-specific (e.g., "biodiversity") receive a high IDF weight.

---

## Score Construction

After obtaining the per-word TF-IDF weights, scores are aggregated by topic:

| Variable | Calculation |
|---|---|
| `esg_environmental` | Sum of TF-IDF weights for all **Environmental** words in the document |
| `esg_social` | Sum of TF-IDF weights for all **Social** words in the document |
| `esg_governance` | Sum of TF-IDF weights for all **Governance** words in the document |
| `esg_total` | `esg_environmental + esg_social + esg_governance` |

This is computed for each segment, yielding eight score columns in total:

| Column | Segment |
|---|---|
| `pres_esg_total` | Presentation |
| `pres_esg_environmental` | Presentation |
| `pres_esg_social` | Presentation |
| `pres_esg_governance` | Presentation |
| `ans_esg_total` | Q&A Answers |
| `ans_esg_environmental` | Q&A Answers |
| `ans_esg_social` | Q&A Answers |
| `ans_esg_governance` | Q&A Answers |

---

## Interpretation

- A score of **0** means no ESG dictionary words appeared in that segment.
- **Higher scores** indicate more ESG-related language, weighted by how distinctive (rare) each word is across the full corpus.
- Scores are **not bounded**: they grow with both the frequency and the breadth of ESG vocabulary used.
- Because IDF downweights common words, a call using many *different* ESG words scores higher than one that repeats the same word many times.
- Scores are **not directly comparable across segments** (presentation vs. Q&A) because text lengths differ systematically — presentations tend to be longer.

---

## Additional Columns in `esg_scores.csv`

| Column | Description |
|---|---|
| `permco` | Compustat/CRSP firm identifier (permanent company number) |
| `permno` | CRSP security identifier (permanent number) |
| `gvkey` | Compustat firm identifier |
| `comnam` | Company name |
| `filename` | Transcript file name (without `.txt`) |
| `date_call` | Date of the earnings call |
| `year_call` | Year of the call |
| `month_call` | Month of the call |
| `day_call` | Day of the call |
| `pres_found` | Boolean — was a presentation segment file found? |
| `ans_found` | Boolean — was a Q&A answers segment file found? |

---

## Citation

Baier, P., Berninger, M., & Kiesel, F. (2020). Environmental, social and governance reporting in annual reports: A textual analysis. *Financial Markets, Institutions & Instruments*, 29(3), 93–118. https://doi.org/10.1111/fmii.12132

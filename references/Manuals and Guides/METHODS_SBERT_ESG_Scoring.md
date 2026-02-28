# ESG Scoring of Earnings-Call Transcripts — Methodology

*Document version: 1.0 — February 2026*
*Author: Bastian Koch*

---

## 1. Overview

This document describes the complete pipeline used to assign Environmental (E),
Social (S), and Governance (G) relevance scores to earnings-call transcripts.
The approach is based on **semantic similarity** between a call transcript and
three authoritative reference corpora, measured via **Sentence-BERT embeddings**.

The key intuition is straightforward: if the language in an earnings call is
semantically close to the language used by expert bodies to describe, say,
climate change and environmental risks (the E corpus), the call receives a high
E score — not because it contains a predefined list of keywords, but because its
overall meaning maps into the same region of the semantic embedding space.

---

## 2. Reference Corpora

### 2.1 Sources

Three reference corpora were assembled, each corresponding to one ESG dimension:

| Dimension | Source document | Pages | Content lines (after cleaning) |
|-----------|----------------|-------|--------------------------------|
| **E — Environmental** | IPCC AR6 Synthesis Report, Full Volume (PDF, 2023) | 186 | 3,688 |
| **E — Environmental** | IPCC AR6 SYR Longer Report (scraped from IPCC website) | — | 836 |
| **S — Social** | OECD Guidelines for Multinational Enterprises (PDF, 2023) | 79 | 1,573 |
| **G — Governance** | G20/OECD Principles of Corporate Governance (PDF, 2023) | 53 | 1,067 |

The two E documents were concatenated into a single E corpus (687,065 characters
total). The S and G corpora are each a single document (245,962 and 183,258
characters, respectively).

### 2.2 Why these sources?

- **IPCC AR6 (E):** The most authoritative and comprehensive current synthesis of
  climate-science findings. It covers physical climate risks, mitigation,
  adaptation, and vulnerability — the full scope of what environmental relevance
  in corporate communications should reflect.

- **OECD Guidelines for MNEs (S):** Covers responsible business conduct toward
  employees, human rights, labour standards, consumer protection, and supply
  chains — all core dimensions of the Social pillar in ESG.

- **OECD/G20 Corporate Governance Principles (G):** The main international
  benchmark for board structure, shareholder rights, transparency, executive
  accountability, and stakeholder engagement — exactly the Governance pillar.

### 2.3 Corpus cleaning

Each PDF was extracted using PyMuPDF (fitz) and the web-scraped IPCC text was
obtained via the `requests` + `BeautifulSoup` pipeline in `src/ipcc_scraper.py`.

Every corpus was then passed through a shared `normalize_for_nlp()` function
(`src/{ipcc,oecd,governance}_pdf_extractor.py`, `src/ipcc_scraper.py`) that:

1. Removes all **digits** (years, percentages, section numbers, table data).
2. Removes **special characters** — degree signs (°), percent (%), ±, ≤, ≥, ×,
   em-dashes, arrows, and similar symbols.
3. Strips all **non-ASCII** characters.
4. Retains only `[a-zA-Z .'-:,;!?()]`.
5. Drops any line shorter than **3 words** (catches figure labels, header
   remnants, axis ticks).

Additional document-specific cleaning was applied (running headers, table of
contents, glossary/back-matter, footnote annotations, figure captions, etc.).
The raw input files were **never modified**; all cleaning happens in memory
during the extraction scripts.

---

## 3. Building the Reference Vectors

*Script:* `src/build_esg_vectors.py`
*Output:* `data/processed/esg_vectors/{E,S,G}_vector.npy`

### 3.1 Model

All text is embedded with **`sentence-transformers/all-mpnet-base-v2`** — a
768-dimension bi-encoder trained on a large mix of semantic similarity tasks.
It was chosen for its strong performance on sentence- and passage-level
similarity benchmarks while remaining tractable on CPU.

GPU is used automatically when available (`torch.cuda.is_available()`);
otherwise the pipeline runs on CPU.

### 3.2 Chunking

Because `all-mpnet-base-v2` has a maximum token length of 384 tokens, long
documents must be split before encoding. A sentence-boundary-aware chunker
(shared by both the reference-vector builder and the scoring script) splits text
into windows of **approximately 1,200–1,800 characters**:

1. The text is first tokenised into crude sentences by splitting on `. `, `! `,
   `? `, and newlines.
2. Sentences are accumulated into a buffer until the buffer reaches
   ~1,200 characters; at that point the buffer is emitted as a chunk and reset.
3. If adding the next sentence would exceed 1,800 characters the buffer is
   emitted first; if the buffer is still below 1,200 the overshoot is accepted
   and hard-cut at 1,800 to avoid producing tiny fragments.

**No stopwords are removed; no stemming or lemmatisation is applied.** Natural
sentence structure is fully preserved so the encoder can exploit syntactic and
contextual signals.

| Corpus | Character length | Chunks |
|--------|-----------------|--------|
| E (IPCC Full Volume + Longer Report) | 687,065 | 541 |
| S (OECD Guidelines) | 245,962 | 194 |
| G (OECD Corporate Governance) | 183,258 | 144 |

### 3.3 Encoding and mean-pooling

Each chunk is encoded with `model.encode(..., normalize_embeddings=True)`,
which L2-normalises every chunk vector to unit length immediately after the
forward pass. The chunk embeddings are then **mean-pooled** across all chunks
in the corpus:

$$\bar{v} = \frac{1}{N} \sum_{i=1}^{N} e_i$$

The mean vector is then **re-normalised** to unit length:

$$v_{\text{ref}} = \frac{\bar{v}}{\|\bar{v}\|}$$

This yields one 768-dimensional unit vector per ESG dimension.

### 3.4 Reference vector diagnostics

Pairwise cosine similarity between the three reference vectors:

| Pair | Cosine similarity |
|------|------------------|
| cos(**E**, **S**) | 0.437 |
| cos(**E**, **G**) | 0.264 |
| cos(**S**, **G**) | 0.766 |

The E vector is the most distinct (climate science language differs substantially
from corporate governance language). The relatively high S/G similarity reflects
the shared institutional OECD vocabulary in those two documents (both address
corporate responsibility, stakeholder relations, and regulatory frameworks).
The separation between all three vectors confirms that they capture meaningfully
different semantic regions, which is a prerequisite for the scores to be
informative.

---

## 4. Scoring Earnings-Call Transcripts

*Script:* `src/score_transcripts_sbert.py`
*Output:* `results/esg_call_similarity.{parquet,csv}`

### 4.1 Input: processed transcript segments

The earnings-call processing pipeline (`src/earnings_calls_processing.py`)
pre-segments each transcript into three files stored in
`data/processed/Transcripts/Call_segments/`:

| File suffix | Content |
|-------------|---------|
| `_presentation.txt` | Management prepared remarks (operator/speaker headers removed) |
| `_answers.txt` | Management answers in the Q&A session |
| `_questions.txt` | Analyst questions in the Q&A session |

For scoring, three **text surfaces** are constructed:

| Surface | Construction | Score column prefix |
|---------|-------------|---------------------|
| **Whole call** | presentation + answers + questions | `E_call`, `S_call`, `G_call` |
| **Presentation** | presentation only | `pres_E_call`, `pres_S_call`, `pres_G_call` |
| **Q&A** | answers + questions | `qa_E_call`, `qa_S_call`, `qa_G_call` |

### 4.2 Batch encoding strategy

Rather than calling `model.encode()` separately for each of the ~5,958 calls
(which would incur Python-side overhead ~18,000 times), the pipeline
**collects all chunks from all calls and all surfaces into a single flat list**
and encodes them in one forward pass. This is substantially faster in practice.

For each call row $i$ and each surface $s$, the index range
$(start_{i,s},\ end_{i,s})$ into the flat chunk array is stored. After encoding,
the slice `embeddings[start:end]` is retrieved for each (call, surface) pair
and mean-pooled.

Total chunks across all 5,958 calls and 3 surfaces: approximately 215,000.

### 4.3 Computing the ESG similarity score

For each (call, surface) pair, after mean-pooling and re-normalising:

$$v_{\text{call}} = \frac{\bar{e}_{\text{call}}}{\|\bar{e}_{\text{call}}\|}$$

The score for dimension $d \in \{E, S, G\}$ is the **cosine similarity** between
the call vector and the reference vector:

$$\text{score}_{d} = \cos(v_{\text{call}},\ v_{\text{ref},d}) = v_{\text{call}} \cdot v_{\text{ref},d}$$

Because both vectors are unit-length, cosine similarity equals the dot product.
The resulting scores lie in $(-1, 1)$, with higher values indicating greater
semantic proximity to the corresponding ESG corpus. In practice scores range
from roughly 0.10 to 0.45 across the full dataset (see Section 5).

---

## 5. Output Dataset and Score Interpretation

### 5.1 Output file

`results/esg_call_similarity.parquet` (also `.csv`), one row per earnings call.

| Column | Description |
|--------|-------------|
| `permco`, `permno`, `gvkey`, `comnam` | CRSP / Compustat firm identifiers |
| `filename` | Source transcript filename |
| `date_call`, `year_call`, `month_call`, `day_call` | Call date |
| `presentation_found`, `q&a_found` | Flags from the processing step |
| `E_call` | Whole-call environmental similarity score |
| `S_call` | Whole-call social similarity score |
| `G_call` | Whole-call governance similarity score |
| `pres_E_call` | Presentation environmental similarity score |
| `pres_S_call` | Presentation social similarity score |
| `pres_G_call` | Presentation governance similarity score |
| `qa_E_call` | Q&A environmental similarity score |
| `qa_S_call` | Q&A social similarity score |
| `qa_G_call` | Q&A governance similarity score |

5,958 calls from 2003 to 2024.

### 5.2 Score distributions (whole-call scores, full dataset)

|  | E\_call | S\_call | G\_call |
|--|---------|---------|---------|
| **Mean** | 0.206 | 0.326 | 0.300 |
| **Std dev** | 0.056 | 0.043 | 0.052 |
| **p10** | 0.143 | 0.271 | 0.241 |
| **p25** | 0.169 | 0.299 | 0.270 |
| **p50** | 0.200 | 0.327 | 0.300 |
| **p75** | 0.236 | 0.354 | 0.331 |
| **p90** | 0.273 | 0.380 | 0.364 |

The **E** scores are systematically lower and more dispersed than S and G.
This reflects the fact that climate-science vocabulary (used in the IPCC source)
is far more specialised and less common in standard corporate earnings language
than the governance/social policy language of the OECD sources. A high E score
is therefore a more discriminating signal: it specifically identifies calls where
management uses language close to that of climate-risk and sustainability science.

**S** scores are on average the highest. General business, labour, and
stakeholder language overlaps substantially with OECD Guidelines vocabulary,
so even calls that do not focus on social topics achieve moderate S scores.
This means S scores are best interpreted in relative (cross-firm or over-time)
terms rather than as an absolute indicator.

**G** scores occupy an intermediate position. Governance concepts (board, risk,
accountability, compliance, shareholders) appear regularly in earnings calls,
but the OECD Principles source also uses specific policy and regulatory language
that not all corporate speakers employ.

### 5.3 Cross-score correlations (whole-call scores)

|  | E\_call | S\_call | G\_call |
|--|---------|---------|---------|
| **E\_call** | 1.000 | 0.531 | 0.244 |
| **S\_call** | 0.531 | 1.000 | 0.728 |
| **G\_call** | 0.244 | 0.728 | 1.000 |

The low E–G correlation (0.244) confirms that the environmental score captures
genuinely distinct variation from the governance score. The high S–G correlation
(0.728) is consistent with the similarity of the S and G reference vectors
themselves (cos = 0.766): both OECD sources share institutional vocabulary.
Researchers should be aware of this when using S and G scores jointly in
regression analyses; including both may introduce multicollinearity.

### 5.4 Missing values

| Column | NaNs | Reason |
|--------|------|--------|
| `E/S/G_call` | 2 | Calls where no segment files were found at all |
| `pres_E/S/G_call` | 14 | Calls flagged as having no presentation segment |
| `qa_E/S/G_call` | 95 | Calls with no Q&A segment (presentation-only calls) |

---

## 6. Design Choices and Limitations

### 6.1 Choice of embedding model

`all-mpnet-base-v2` was selected because it (a) is publicly available and
reproducible without API keys, (b) has strong performance on the SBERT
benchmarks for semantic similarity, and (c) is feasible on CPU in under an hour
for the full dataset. The model was used with default tokenisation and no
fine-tuning; it was not adapted specifically to financial or ESG text. Users
who require domain-adapted representations may consider financial pre-trained
models (e.g., FinBERT), though these would require re-building the reference
vectors with the same model.

### 6.2 Mean-pooling vs. max-pooling or full-document encoding

Mean-pooling of chunk vectors produces a central tendency estimate of the
document's semantic content. An alternative would be to take the maximum
cosine similarity across chunks (i.e., the closest passage to the reference),
which would capture whether any *part* of the call touches on ESG topics,
rather than the document-average. The current choice (mean) is more
appropriate for characterising the overall ESG orientation of the call.

### 6.3 Corpus breadth vs. specificity

More specific or narrower reference corpora (e.g., only the physical-risk
chapter of the IPCC report for E) would yield sharper discrimination but higher
sensitivity to the exact source selection. The full-volume IPCC source was
favoured to capture the full scope of environmental relevance (mitigation,
adaptation, loss and damage, governance of climate action).

### 6.4 No stopword removal or stemming

The decision to preserve raw transcript text (no stopword removal, no stemming)
means the embeddings benefit from the full syntactic context that the transformer
encoder was trained on. Stripping stopwords or stemming is beneficial for
count-based methods (TF-IDF) but generally harmful for contextual embedding
models.

### 6.5 Relationship to the TF-IDF ESG scores

The project also contains TF-IDF-based ESG scores (`data/processed/esg_scores.csv`)
computed from the Baier/Berninger/Kiesel (2020) ESG word list. The Sentence-BERT
scores in `results/esg_call_similarity.parquet` are methodologically
complementary:

| Aspect | TF-IDF / word-list | Sentence-BERT |
|--------|--------------------|---------------|
| Basis | Curated ESG vocabulary | Semantic corpora |
| Context sensitivity | None (bag of words) | Full sentence context |
| Out-of-vocabulary | Misses synonyms | Handles paraphrase |
| Interpretability | High (word counts) | Lower (embedding space) |
| Speed | Very fast | Slow (GPU/CPU inference) |

The two score families can be used in combination (e.g., as independent
variables in a panel regression) or compared to assess robustness.

---

## 7. Reproducibility

The full pipeline is deterministic (random seed = 42 in all scripts).

To reproduce from scratch:

```bash
# 1. Build the reference text corpus
make build_corpus

# 2. Build ESG reference vectors
make build_esg_vectors

# 3. Score all transcripts
make score_sbert
```

Or in a single chain:

```bash
make score_sbert   # depends on build_esg_vectors which depends on build_corpus
```

Key file locations:

| Artefact | Path |
|----------|------|
| E corpus | `data/processed/Text corpus/E/` |
| S corpus | `data/processed/Text corpus/S/` |
| G corpus | `data/processed/Text corpus/G/` |
| Reference vectors | `data/processed/esg_vectors/{E,S,G}_vector.npy` |
| Transcript segments | `data/processed/Transcripts/Call_segments/` |
| Scores (full dataset) | `results/esg_call_similarity.{parquet,csv}` |

Python environment: `venv/` (Python 3.10). Key packages: `sentence-transformers`,
`torch`, `numpy`, `pandas`, `pyarrow`, `pymupdf`, `beautifulsoup4`, `tqdm`.
See `requirements.txt` for the full list.

---

## 8. References

- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using
  Siamese BERT-networks. *EMNLP 2019*.
  https://doi.org/10.18653/v1/D19-1410

- IPCC (2023). *AR6 Synthesis Report: Climate Change 2023*.
  Intergovernmental Panel on Climate Change, Geneva.
  https://www.ipcc.ch/report/ar6/syr/

- OECD (2023). *OECD Guidelines for Multinational Enterprises on Responsible
  Business Conduct*. OECD Publishing, Paris.
  https://doi.org/10.1787/81f92357-en

- OECD (2023). *G20/OECD Principles of Corporate Governance*.
  OECD Publishing, Paris.
  https://doi.org/10.1787/ed750b30-en

- Baier, P., Berninger, M., & Kiesel, F. (2020). Environmental, social and
  governance in sovereign bond spreads. *Journal of Asset Management*, 21(4), 
  333–349. (ESG word list used for TF-IDF baseline scores.)

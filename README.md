# Textual Analysis Group Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A group project for a Textual Analysis course that analyzes earnings call transcripts using natural language processing techniques. This dataset contains corporate earnings conference call transcripts spanning multiple years and companies.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Supplementary third-party data (e.g., stock prices, financials)
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- Primary dataset in original form (e.g., earnings call transcripts)
│
├── references         <- Data dictionaries, manuals, and all explanatory materials
│
├── output
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── tables         <- Generated LaTeX and text tables with summary statistics
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
└── code               <- Source code for the analysis pipeline
    │
    ├── __init__.py                          <- Makes code a Python module
    │
    │   ── 0. Data Preparation ──
    ├── 0.1_extract_gvkeys.py                <- Extract firm identifiers (gvkey, permno)
    ├── 0.2_earnings_calls_processing.py     <- NLP normalization and transcript segmentation
    ├── 0.3_governance_pdf_extractor.py      <- Extract G corpus from OECD PDF
    ├── 0.4_ipcc_pdf_extractor.py            <- Extract E corpus from IPCC PDF
    ├── 0.5_oecd_pdf_extractor.py            <- Extract S corpus from OECD Guidelines PDF
    ├── 0.6_corpora_processing.py            <- Normalize E/S/G reference corpora
    ├── 0.7_LMD_frac.py                      <- Compute Loughran-McDonald word fractions
    │
    │   ── 1. Feature Engineering ──
    ├── 1.1_build_earnings_calls_bigrams.py  <- Build bigrams from transcript segments
    ├── 1.2_build_tfidf.py                   <- Build TF-IDF matrices for transcripts
    ├── 1.3_build_corpus_esg_vectors.py      <- Project ESG corpora into TF-IDF space
    ├── 1.4_compute_esg_talk.py              <- Compute ESG cosine-similarity talk scores
    │
    │   ── 2. Exploration ──
    ├── 2.1_bigram_exploration.py            <- Top-15 bigrams per ESG corpus (bar charts)
    ├── 2.2_esg_talk_exploration.py          <- ESG talk distributions and time trends
    │
    │   ── 3. Analysis ──
    ├── 3.1_event_study.py                   <- CAR event study and panel regressions
    └── 3.2_esg_ratings.py                   <- ESG talk vs. next-year MSCI ESG ratings
```

## Pipeline Overview

Run scripts in numbered order. Each script reads from `data/` and writes outputs back to `data/` or `output/`.

```
0.x  →  data preparation & corpus extraction
1.x  →  bigram / TF-IDF / ESG talk feature engineering
2.x  →  exploratory plots  →  output/figures/
3.x  →  regressions / event study  →  output/tables/
```

## Getting Started

Follow these steps to set up the project on your local machine for the first time.

### Prerequisites

- Python 3.10+ installed on your system
- Git installed
- Access to the project data (see [Data Access](#data-access) below)

### 1. Clone the Repository

```bash
git clone https://github.com/BastianKoch/Textual-analysis-project.git
cd Textual-analysis-project
```

### 2. Create a Virtual Environment

Create an isolated Python environment for this project:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Or on Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

Install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Get the Data

The earnings call transcripts are stored in a shared location (not in this repository due to file size).

**Download the data:**
- Request access from the group members or download from: [Shared Dropbox folder link]
- Extract the transcript files to: `data/raw/Transcripts/`

Your structure should look like:
```
data/raw/Transcripts/
├── 2003/
│   ├── 612460.txt
│   ├── 638281.txt
│   └── ...
├── 2004/
├── 2005/
└── ...
```

### 5. Run a Script

```bash
# Make sure your venv is activated
python code/1.4_compute_esg_talk.py
```

### 6. Working with the Code

**To run a script from the project root:**

```bash
python code/0.2_earnings_calls_processing.py
```

**To add more packages:**

```bash
pip install <package-name>
```

Then add it to `requirements.txt` so teammates can install it too.

### Deactivating the Environment

When you're done working:

```bash
deactivate
```

## Data Access

The earnings call transcript data is not included in this repository. To access it:

1. **Dropbox Link**: [Add your shared Dropbox link here]
2. **Alternative**: Contact the group members for access
3. **Place data here**: `data/raw/Transcripts/` (create the folder if it doesn't exist)

## Contributing

1. Create a new branch for your work: `git checkout -b feature/your-feature-name`
2. Make your changes and test them
3. Commit with clear messages: `git commit -m "Add feature description"`
4. Push to GitHub: `git push origin feature/your-feature-name`
5. Create a Pull Request for review

--------


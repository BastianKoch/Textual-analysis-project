# Textual Analysis Group Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Textual analysis class group project

## Project Organization

```
├── Makefile           <- Makefile with convenience commands (e.g., `make requirements`)
├── README.md          <- The top-level README for developers using this project
├── data
│   ├── external       <- Supplementary third-party data (e.g., stock prices, financials)
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   └── raw            <- Primary dataset in original form (e.g., earnings call transcripts)
│
├── literature         <- Literature, data dictionaries, and all explanatory materials
│
├── notebooks          <- Jupyter notebooks for exploration and analysis
│                         Naming convention: number-initials-description
│                         (e.g., `1.0-bk-initial-data-exploration`)
│
├── pyproject.toml     <- Project configuration file with package metadata and tool settings
│
├── reports            
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment
│
├── reports            
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── tables         <- Generated tables and summary statistics
│
└── src                <- Source code for use in this project
    │
    ├── __init__.py    <- Makes src a Python module
    │
    └── (your reusable code, functions, and classes go here)
```

## Notebooks vs Source Code

- **notebooks/**: Use for exploration, analysis, and one-off experiments. Import functions from `src/`.
- **src/**: Put reusable functions, classes, and data processing pipelines here. Import these into your notebooks.

## Getting Started

Install dependencies:
```bash
make requirements
# or: pip install -r requirements.txt
```

--------


# Textual Analysis Group Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A group project for a Textual Analysis course that analyzes earnings call transcripts using natural language processing techniques. This dataset contains corporate earnings conference call transcripts spanning multiple years and companies.

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
├── references         <- Data dictionaries, manuals, and all explanatory materials
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

### 4. Set Up Jupyter Kernel (Optional, for notebooks)

Register this virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name=ta_project --display-name="Python 3.10 (ta_project)"
```

Then in VS Code or Jupyter, select this kernel when running notebooks.

### 5. Get the Data

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

### 6. Run Your First Notebook

Start Jupyter and open the cleaning notebook:

```bash
# Make sure your venv is activated
jupyter notebook
```

Then open `1.0-process-transcripts.ipynb` and run through the cells to:
1. Load and inspect a sample transcript
2. Test the cleaning functions
3. Process all transcripts

### 7. Working with the Code

**To use functions from the `src/` module in your notebook:**

```python
from src.text_processing import load_transcript, clean_text
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


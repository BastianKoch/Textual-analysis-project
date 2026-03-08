"""
ESG Talk and ESG Scores — Panel Regressions
============================================

Merge ESG talk scores with MSCI ESG ratings and run fixed-effects panel
regressions examining whether ESG-related language in earnings calls
predicts next-year ESG scores.

Input
-----
    data/external/msci_esg_ratings.csv
    data/external/compustat_annual_2002_2024.csv
    data/external/crsp_daily_2002_2024.csv
    data/processed/esg_talk.csv
    data/processed/LMD_Frac.csv

Output
------
    output/tables/summary_statistics_ESG_ratings.tex
    output/tables/summary_statistics_ESG_ratings.txt
    output/tables/regression_table_ESG_ratings.tex
    output/tables/regression_table_ESG_ratings.txt

Usage
-----
    python code/3.2_esg_ratings.py

Authors
-------
    Daniel
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from linearmodels import PanelOLS
from pathlib import Path
from stargazer.stargazer import Stargazer

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[1]
EXTERNAL   = ROOT / "data" / "external"
PROCESSED  = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "output" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

ESG_RATINGS_FILE = EXTERNAL  / "msci_esg_ratings.csv"
ESG_TALK_FILE    = PROCESSED / "esg_talk.csv"
COMPUSTAT_FILE   = EXTERNAL  / "compustat_annual_2002_2024.csv"
CRSP_FILE        = EXTERNAL  / "crsp_daily_2002_2024.csv"
LMD_FRAC_FILE    = PROCESSED / "LMD_Frac.csv"

CONTROLS       = ['NEGFRAC', 'LITFRAC', 'UNCFRAC', 'LOGWORD',
                  'rev_change', 'leverage', 'ln_btm', 'ln_size', 'roa', 'capex_scaled']
DEPENDENT_VARS = ['env_score_next_year', 'soc_score_next_year',
                  'gov_score_next_year', 'esg_score_next_year']
INDEP_VARS_PRES     = ['e_talk_pres',     's_talk_pres',     'g_talk_pres']
INDEP_VARS_ANSWERS  = ['e_talk_answers',  's_talk_answers',  'g_talk_answers']
INDEP_VARS_COMBINED = ['e_talk_combined', 's_talk_combined', 'g_talk_combined']
ALL_INDEP_VARS      = INDEP_VARS_PRES + INDEP_VARS_ANSWERS + INDEP_VARS_COMBINED

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("Loading data...")
esg_ratings = pd.read_csv(ESG_RATINGS_FILE)
esg_talk    = pd.read_csv(ESG_TALK_FILE)
compustat   = pd.read_csv(COMPUSTAT_FILE)
crsp        = pd.read_csv(CRSP_FILE)
lmd_frac    = pd.read_csv(LMD_FRAC_FILE)
print(f"  ESG ratings : {len(esg_ratings):,} rows")
print(f"  ESG talk    : {len(esg_talk):,} rows")
print(f"  Compustat   : {len(compustat):,} rows")
print(f"  CRSP        : {len(crsp):,} rows")
print(f"  LMD frac    : {len(lmd_frac):,} rows")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPARE DATES
# ─────────────────────────────────────────────────────────────────────────────
print("\nPreparing dates...")

esg_talk['date'] = pd.to_datetime(
    esg_talk[['year_call', 'month_call', 'day_call']]
    .astype(str).agg('-'.join, axis=1)
)

esg_ratings['date'] = pd.to_datetime(esg_ratings['date'])

lmd_frac['date'] = pd.to_datetime(
    lmd_frac['year_call'].astype(str) + '-' +
    lmd_frac['month_call'].astype(str) + '-' +
    lmd_frac['date_call'].astype(str).str[:2]
)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FIRM CHARACTERISTICS (Compustat + CRSP)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCalculating firm characteristics...")

# Adjust fiscal year
compustat['fyear'] = compustat['fyear'] + 1
compustat = compustat.sort_values(['gvkey', 'fyear'])

compustat['rev_change']    = compustat.groupby('gvkey')['revt'].pct_change()
compustat['leverage']      = (compustat['dltt'] + compustat['dlc']) / compustat['at']
compustat['roa']           = compustat['ni'] / compustat['at']
compustat['capex_scaled']  = compustat['capx'] / compustat['at']
compustat.replace([np.inf, -np.inf], np.nan, inplace=True)

crsp = crsp[crsp['SHRCD'].isin([10, 11]) & crsp['EXCHCD'].isin([1, 2, 3])].copy()
crsp['mktcap']  = crsp['SHROUT'] * crsp['PRC']
crsp['ln_size'] = np.log(crsp['mktcap'])
crsp.rename(columns={'PERMNO': 'permno'}, inplace=True)
crsp['date'] = pd.to_datetime(crsp['date'])

# ─────────────────────────────────────────────────────────────────────────────
# 4.  SUMMARY STATISTICS TABLE  (esg_talk + lmd_frac only, all observations)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCreating summary statistics table...")

lmd_frac_stats = lmd_frac['date']   # already parsed above
merged_stats = pd.merge(esg_talk, lmd_frac, on=['permno', 'date'], how='inner')

lmd_vars  = ['NEGFRAC', 'LITFRAC', 'UNCFRAC', 'LOGWORD']
stats_df  = merged_stats[ALL_INDEP_VARS + lmd_vars].describe().T
stats_df['N'] = merged_stats[ALL_INDEP_VARS + lmd_vars].count()
stats_df = stats_df[['N', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

VAR_LABELS = {
    'e_talk_pres':     'E Talk (Presentation)',
    's_talk_pres':     'S Talk (Presentation)',
    'g_talk_pres':     'G Talk (Presentation)',
    'e_talk_answers':  'E Talk (Q\\&A)',
    's_talk_answers':  'S Talk (Q\\&A)',
    'g_talk_answers':  'G Talk (Q\\&A)',
    'e_talk_combined': 'E Talk (Combined)',
    's_talk_combined': 'S Talk (Combined)',
    'g_talk_combined': 'G Talk (Combined)',
    'NEGFRAC':         'Negative Word Fraction',
    'LITFRAC':         'Litigation Word Fraction',
    'UNCFRAC':         'Uncertainty Word Fraction',
    'LOGWORD':         'Log(Word Count)',
}

def _stats_row(var):
    r = stats_df.loc[var]
    return (f"{VAR_LABELS.get(var, var)} & {int(r['N'])} & "
            f"{r['mean']:.3f} & {r['std']:.3f} & {r['min']:.3f} & "
            f"{r['25%']:.3f} & {r['50%']:.3f} & {r['75%']:.3f} & {r['max']:.3f} \\\\")

latex_stats  = "\\begin{table}[htbp]\n\\centering\n"
latex_stats += "\\caption{Summary Statistics: ESG Talk and Linguistic Variables}\n"
latex_stats += "\\label{tab:summary_stats}\n"
latex_stats += "\\begin{tabular}{lcccccccc}\n\\hline\\hline\n"
latex_stats += "Variable & N & Mean & Std Dev & Min & 25\\% & Median & 75\\% & Max \\\\\n\\hline\n"
latex_stats += "\\multicolumn{9}{l}{\\textit{Panel A: ESG Talk Variables}} \\\\\n"
for v in ALL_INDEP_VARS:
    if v in stats_df.index:
        latex_stats += _stats_row(v) + "\n"
latex_stats += "\\hline\n"
latex_stats += "\\multicolumn{9}{l}{\\textit{Panel B: Loughran-McDonald Linguistic Variables}} \\\\\n"
for v in lmd_vars:
    if v in stats_df.index:
        latex_stats += _stats_row(v) + "\n"
latex_stats += ("\\hline\\hline\n\\end{tabular}\n"
                "\\begin{tablenotes}\\small\n"
                "\\item ESG Talk variables measure cosine similarity of earnings call text to ESG corpora. "
                "Loughran-McDonald variables capture negative, litigation, and uncertainty language.\n"
                "\\end{tablenotes}\n\\end{table}\n")

out_stats = TABLES_DIR / "summary_statistics_ESG_ratings.tex"
out_stats.write_text(latex_stats, encoding="utf-8")
print(f"  Saved → output/tables/summary_statistics_ESG_ratings.tex")

out_stats_txt = TABLES_DIR / "summary_statistics_ESG_ratings.txt"
out_stats_txt.write_text(latex_stats, encoding="utf-8")
print(f"  Saved → output/tables/summary_statistics_ESG_ratings.txt")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MERGE DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("\nMerging datasets...")

merged = pd.merge(esg_ratings, esg_talk, on=['permno', 'date'], how='inner')
merged = pd.merge(merged, compustat, left_on=['gvkey', 'year_call'],
                  right_on=['gvkey', 'fyear'], how='inner')
merged = pd.merge(merged, lmd_frac, on=['permno', 'date'], how='inner')
merged = pd.merge(merged, crsp,     on=['permno', 'date'], how='inner')

merged['btm']    = merged['ceq'] / merged['mktcap']
merged['ln_btm'] = np.log(merged['btm'])
merged['industry'] = merged['SICCD'].astype(str).str[:2]

print(f"  Merged shape: {merged.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PREPARE PANEL DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\nPreparing panel data...")

merged = merged.set_index(['permno', 'date'])

required_cols = CONTROLS + DEPENDENT_VARS + ALL_INDEP_VARS + ['date_call_x', 'industry']
merged = merged[required_cols].dropna()

# Normalise ESG talk variables
for var in ALL_INDEP_VARS:
    merged[var] = (merged[var] - merged[var].mean()) / merged[var].std()

print(f"  Final regression sample: {len(merged):,} observations")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  REGRESSIONS
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning regressions...")

industry_dummies = pd.get_dummies(merged['industry'], prefix='ind', drop_first=True)

MODEL_SPECS = [
    ('e_talk_pres',     'env_score_next_year'),
    ('s_talk_pres',     'soc_score_next_year'),
    ('g_talk_pres',     'gov_score_next_year'),
    (INDEP_VARS_PRES,   'esg_score_next_year'),
    ('e_talk_answers',  'env_score_next_year'),
    ('s_talk_answers',  'soc_score_next_year'),
    ('g_talk_answers',  'gov_score_next_year'),
    (INDEP_VARS_ANSWERS,'esg_score_next_year'),
    ('e_talk_combined', 'env_score_next_year'),
    ('s_talk_combined', 'soc_score_next_year'),
    ('g_talk_combined', 'gov_score_next_year'),
    (INDEP_VARS_COMBINED,'esg_score_next_year'),
]

results_list = []
for indep_vars, dep_var in MODEL_SPECS:
    if isinstance(indep_vars, str):
        indep_vars = [indep_vars]
    X = pd.concat([merged[indep_vars + CONTROLS], industry_dummies], axis=1)
    y = merged[dep_var]
    model  = PanelOLS(y, X, time_effects=True, drop_absorbed=True)
    result = model.fit(cov_type='clustered', cluster_entity=True)
    results_list.append(result)
    print(f"  ✓ {dep_var}  ~  {indep_vars[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  REGRESSION TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\nBuilding regression table...")

star = Stargazer(results_list)
star.custom_columns(
    ['Env (t+1)', 'Soc (t+1)', 'Gov (t+1)', 'ESG (t+1)'] * 3,
    [1] * 12
)
star.title('Fixed Effects Regressions: Next-Year ESG Scores')
star.add_line('Year FE',        ['Yes'] * 12)
star.add_line('Industry FE',    ['Yes'] * 12)
star.add_line('ESG Talk',
              ['Presentation'] * 4 + ['Q\\&A'] * 4 + ['Combined'] * 4)
star.show_degrees_of_freedom(False)
star.show_model_numbers(False)
star.add_custom_notes([
    'Dependent variables are next-year ESG scores. SE clustered at firm level.',
    '*** p<0.01, ** p<0.05, * p<0.10',
])

latex_reg = star.render_latex()

out_reg = TABLES_DIR / "regression_table_ESG_ratings.tex"
out_reg.write_text(latex_reg, encoding="utf-8")
print(f"  Saved → output/tables/regression_table_ESG_ratings.tex")

out_reg_txt = TABLES_DIR / "regression_table_ESG_ratings.txt"
out_reg_txt.write_text(latex_reg, encoding="utf-8")
print(f"  Saved → output/tables/regression_table_ESG_ratings.txt")

print("\nDone.")

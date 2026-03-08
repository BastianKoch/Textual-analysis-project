"""
Event Study and Panel Regressions
==================================

Compute Cumulative Abnormal Returns (CAR) around ESG-related earnings
calls using a Fama-French three-factor model, and run panel regressions
with ESG talk and Loughran-McDonald controls.

Input
-----
    data/external/IBES.xlsx
    data/external/crsp_daily_2002_2024.csv
    data/external/F-F_Research_Data_Factors_daily.xlsx
    data/external/compustat_annual_2002_2024.csv
    data/processed/esg_talk.csv
    data/processed/LMD_Frac.csv

Output
------
    output/tables/summary_statistics_ESG_talk.tex
    output/tables/summary_statistics_ESG_talk.txt
    output/tables/regression_models2to5.txt

Usage
-----
    python code/3.1_event_study.py

Authors
-------
    Aryan
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0.  CONFIGURATION 
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

IBES_FILE  = ROOT / "data" / "external" / "IBES.xlsx"
CRSP_FILE  = ROOT / "data" / "external" / "crsp_daily_2002_2024.csv"
ESG_FILE   = ROOT / "data" / "processed" / "esg_talk.csv"
LMD_Frac   = ROOT / "data" / "processed" / "LMD_Frac.csv"
FF_Factors = ROOT / "data" / "external" / "F-F_Research_Data_Factors_daily.xlsx"

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────

print("Loading IBES data...")
ibes = pd.read_excel(IBES_FILE)

# Standardise column names (handle spaces and mixed cases)
ibes.columns = (ibes.columns
                .str.strip()
                .str.lower()
                .str.replace(r'[^a-z0-9]', '_', regex=True)
                .str.replace(r'_+', '_', regex=True)
                .str.strip('_'))

# Rename to working names
ibes_col_map = {
    'cusip':                                             'cusip',
    'ibes_ticker_symbol':                               'ibes_ticker',
    'official_ticker_symbol':                           'ticker',
    'company_name':                                     'company_name',
    'measure':                                          'measure',
    'forecast_period_indicator':                        'fpi',
    'median_estimate':                                  'median_est',
    'mean_estimate':                                    'mean_est',
    'forecast_period_end_date_sas_format':              'fpedats',
    'actual_value_from_the_detail_actuals_file':        'actual',
    'announce_date_of_the_actual_from_the_detail_actuals_file': 'anndats',
}
ibes.rename(columns={k: v for k, v in ibes_col_map.items() if k in ibes.columns},
            inplace=True)

# Parse dates
ibes['anndats'] = pd.to_datetime(ibes['anndats'], errors='coerce')
ibes['fpedats'] = pd.to_datetime(ibes['fpedats'], errors='coerce')
ibes.dropna(subset=['anndats', 'actual', 'mean_est'], inplace=True)

print(f"  IBES rows after filtering: {len(ibes):,}")

# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading CRSP data...")
crsp = pd.read_csv(CRSP_FILE)
crsp.columns = crsp.columns.str.strip().str.upper()

# Rename to working names
crsp_col_map = {
    'PERMNO': 'permno',
    'DATE':   'date',
    'CUSIP':  'cusip',
    'TICKER': 'ticker',
    'PRC':    'prc',
    'RET':    'ret',
    'SICCD':  'siccd',
}
crsp.rename(columns={k: v for k, v in crsp_col_map.items() if k in crsp.columns},
            inplace=True)

crsp['date'] = pd.to_datetime(crsp['date'].astype(str), errors='coerce')
crsp['ret']  = pd.to_numeric(crsp['ret'], errors='coerce')
crsp['prc']  = pd.to_numeric(crsp['prc'].abs() if crsp['prc'].dtype != object
                             else crsp['prc'].str.replace('-','').astype(float),
                             errors='coerce')
crsp.dropna(subset=['date', 'ret', 'prc'], inplace=True)

print(f"  CRSP rows after cleaning: {len(crsp):,}")

# Build 2-digit SIC map (one code per permno)
sic_map = (crsp[['permno', 'siccd']]
           .dropna(subset=['siccd'])
           .drop_duplicates(subset=['permno'], keep='last'))
sic_map['siccd'] = pd.to_numeric(sic_map['siccd'], errors='coerce')
sic_map = sic_map.dropna(subset=['siccd'])
sic_map['sic2'] = sic_map['siccd'].astype(int).astype(str).str.zfill(4).str[:2]
print(f"  SIC map: {len(sic_map):,} permnos with SIC code")

# ─────────────────────────────────────────────────────────────────────────────
print("\nLoading ESG data...")
esg = pd.read_csv(ESG_FILE)
esg.columns = esg.columns.str.strip().str.lower()
esg['date_call'] = pd.to_datetime(esg['date_call'], errors='coerce')
esg['permno'] = pd.to_numeric(esg['permno'], errors='coerce')
print(f"  ESG rows: {len(esg):,}")


print("\nLoading LMD fractions data...")
nl = pd.read_csv(LMD_Frac)
nl.columns = nl.columns.str.strip().str.lower()
nl['date_call'] = pd.to_datetime(nl['date_call'], errors='coerce')
nl['permno'] = pd.to_numeric(nl['permno'], errors='coerce')

print(f"  LMD rows: {len(nl):,}")

# Merge into main ESG dataframe (before the big merge)
esg = esg.merge(nl[['permno', 'date_call', 'negfrac', 'litfrac' , 'uncfrac' , 'logword']], 
                on=['permno', 'date_call'], 
                how='left')



# ─────────────────────────────────────────────────────────────────────────────
# 2.  EARNINGS SURPRISE  =  (Actual - Mean Forecast) / Stock Price
# ─────────────────────────────────────────────────────────────────────────────
print("\nCalculating earnings surprise...")

# We need the stock price on the day BEFORE the announcement to scale
# Merge CRSP price (lagged 1 day) onto IBES by cusip + anndats
price_lookup = (crsp[['cusip', 'date', 'prc']]
                .dropna(subset=['cusip'])
                .sort_values(['cusip', 'date']))
price_lookup['cusip'] = price_lookup['cusip'].astype(str).str.zfill(9).str[:8]

ibes['cusip_8'] = ibes['cusip'].astype(str).str.zfill(9).str[:8]

# Merge price on announcement date (use closest prior trading day within 5 days)
ibes_sorted = ibes.sort_values('anndats')
price_sorted = price_lookup.sort_values('date')

# Use merge_asof for each cusip: match on nearest prior date
ibes_price = pd.merge_asof(
    ibes_sorted[['cusip_8', 'anndats', 'actual', 'mean_est', 'fpedats', 'ibes_ticker']].sort_values('anndats'),
    price_sorted[['cusip', 'date', 'prc']].rename(columns={'cusip': 'cusip_8', 'date': 'anndats', 'prc': 'price_t0'}),
    on='anndats',
    by='cusip_8',
    direction='backward',
    tolerance=pd.Timedelta('5D')
)

ibes_price.dropna(subset=['price_t0'], inplace=True)
ibes_price['price_t0'] = ibes_price['price_t0'].abs()  

# Earnings surprise: scaled by price
ibes_price['earn_surprise'] = (
    (ibes_price['actual'] - ibes_price['mean_est']) / ibes_price['price_t0']
)

print(f"  Earnings surprise calculated for {len(ibes_price):,} firm-quarters")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  CUMULATIVE ABNORMAL RETURN  CAR[0, 1]
# ─────────────────────────────────────────────────────────────────────────────
print("\nCalculating CAR[0,1]...")

# Market return = Mkt-RF + RF from Kenneth French's data library
ff = pd.read_excel(FF_Factors)  
ff.columns = ff.columns.str.strip()

# Parse date (format: 20020122)
ff['date'] = pd.to_datetime(ff['Date'].astype(str), format='%Y%m%d', errors='coerce')

# French data is in percent (e.g. 0.05 means 0.05%) — divide by 100
ff['mkt_ret'] = (ff['Mkt-RF'] + ff['RF']) / 100

ff = ff[['date', 'mkt_ret']].dropna()

# Merge French market return into CRSP
crsp2 = crsp.merge(ff, on='date', how='left')
crsp2['abnormal_ret'] = crsp2['ret'] - crsp2['mkt_ret']

# Sort CRSP by permno and date for lead computation
crsp2 = crsp2.sort_values(['permno', 'date'])

# We link ESG dates to CRSP for CAR computation
# CAR[0,1]: sum of abnormal return on day 0 (call date) and day+1
# Step 1: get CRSP trading dates index per stock
crsp2['date_str'] = crsp2['date'].dt.strftime('%Y%m%d')

# Build a helper: for each (permno, date), get abnormal return for t and t+1
# Use a shifted return (next trading day)
crsp2['abnormal_ret_t1'] = crsp2.groupby('permno')['abnormal_ret'].shift(-1)

# Merge ESG dates to CRSP to get CAR
esg_crsp = esg.merge(
    crsp2[['permno', 'date', 'prc', 'abnormal_ret', 'abnormal_ret_t1']],
    left_on=['permno', 'date_call'],
    right_on=['permno', 'date'],
    how='left'
)

esg_crsp['CAR_0_1'] = (esg_crsp['abnormal_ret'] + esg_crsp['abnormal_ret_t1']) * 100  # in %

print(f"  CAR calculated for {esg_crsp['CAR_0_1'].notna().sum():,} observations")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MERGE EARNINGS SURPRISE INTO ESG-CRSP DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\nMerging earnings surprise...")

# Link ibes_price to ESG via permno
# We need a CUSIP-PERMNO crosswalk — use CRSP cusip for this
cusip_permno = (crsp[['permno', 'cusip']]
                .dropna(subset=['cusip'])
                .drop_duplicates())
cusip_permno['cusip_8'] = cusip_permno['cusip'].astype(str).str.zfill(9).str[:8]

ibes_price2 = ibes_price.merge(cusip_permno[['permno', 'cusip_8']].drop_duplicates(),
                                on='cusip_8', how='left')

# Match earnings announcement (anndats) closest to the earnings call date
# Many earnings calls happen ON the announcement day or within ±5 days
ibes_price2 = ibes_price2.dropna(subset=['permno']).sort_values('anndats')
esg_crsp     = esg_crsp.sort_values('date_call')

# Align permno dtype for merge_asof (esg permno is float64 from to_numeric; crsp is int64)
esg_crsp['permno']   = esg_crsp['permno'].astype('Int64')
ibes_price2['permno'] = ibes_price2['permno'].astype('Int64')

merged = pd.merge_asof(
    esg_crsp.sort_values('date_call'),
    ibes_price2[['permno', 'anndats', 'earn_surprise']].sort_values('anndats'),
    left_on='date_call',
    right_on='anndats',
    by='permno',
    direction='nearest',
    tolerance=pd.Timedelta('10D')
)

print(f"  Merged observations: {len(merged):,}")
print(f"  With earnings surprise: {merged['earn_surprise'].notna().sum():,}")



# ─────────────────────────────────────────────
# WINSORIZE at 1% / 99%
# ─────────────────────────────────────────────
def winsorize(s, lower=0.01, upper=0.99):
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)

merged['earn_surprise'] = winsorize(merged['earn_surprise'])
merged['CAR_0_1']       = winsorize(merged['CAR_0_1'])

print("Winsorization applied to earn_surprise and CAR_0_1 at 1%/99%")

# Merge 2-digit SIC into main dataset
merged = merged.merge(sic_map[['permno', 'sic2']], on='permno', how='left')
print(f"  Missing SIC after merge: {merged['sic2'].isna().sum():,}")


# ─────────────────────────────────────────────
# SUMMARY STATISTICS (before standardization)
# ─────────────────────────────────────────────

summary_vars = [
    'CAR_0_1',
    'earn_surprise',
    'e_talk_combined',    's_talk_combined', 'g_talk_combined', 
    'e_talk_pres', 's_talk_pres', 'g_talk_pres',
    'e_talk_answers',  's_talk_answers',  'g_talk_answers',
    'negfrac', 'litfrac' , 'uncfrac' , 'logword',
]

# Keep only variables that exist in the dataframe
summary_vars = [v for v in summary_vars if v in merged.columns]

rows = []
for var in summary_vars:
    s = merged[var].dropna()
    rows.append({
        'Variable': var,
        'N':        int(s.count()),
        'Mean':     round(s.mean(), 4),
        'SD':       round(s.std(), 4),
        'Minimum':  round(s.min(), 4),
        'Median':   round(s.median(), 4),
        'Maximum':  round(s.max(), 4),
    })

summary_df = pd.DataFrame(rows)

# Print to console
print("\n" + "="*75)
print("  SUMMARY STATISTICS ")
print("="*75)
print(summary_df.to_string(index=False))
print("="*75)

# Export to LaTeX and txt
tables_dir = ROOT / "output" / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)
summary_df.to_latex(tables_dir / "summary_statistics_ESG_talk.tex", index=False, float_format="%.4f")
print(f"\nSummary statistics saved to: output/tables/summary_statistics_ESG_talk.tex")

txt_lines = ["=" * 75, "  SUMMARY STATISTICS", "=" * 75,
             summary_df.to_string(index=False), "=" * 75]
(tables_dir / "summary_statistics_ESG_talk.txt").write_text("\n".join(txt_lines), encoding="utf-8")
print(f"Summary statistics saved to: output/tables/summary_statistics_ESG_talk.txt")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  STANDARDISE VARIABLES (mean=0, std=1)
# ─────────────────────────────────────────────────────────────────────────────
esg_vars = [
    'e_talk_combined',    's_talk_combined', 'g_talk_combined', 
    'e_talk_pres', 's_talk_pres', 'g_talk_pres',
    'e_talk_answers',  's_talk_answers',  'g_talk_answers',
    'negfrac' , 'litfrac' , 'uncfrac' , 'logword', 'earn_surprise' , 
]

for col in esg_vars:
    if col in merged.columns:
        merged[f'{col}_std'] = (merged[col] - merged[col].mean()) / merged[col].std()

print("\n variables standardised.")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  PREPARE PANEL DATASET
# ─────────────────────────────────────────────────────────────────────────────

merged['year'] = pd.to_datetime(merged['date_call']).dt.year

# Drop rows missing CAR or key regressors
reg_df = merged.dropna(subset=['CAR_0_1', 'earn_surprise']).copy()



print(f"\nFinal regression sample: {len(reg_df):,} firm-quarter observations")
print(f"  Years: {reg_df['year'].min()} – {reg_df['year'].max()}")
print(f"  Unique firms: {reg_df['permno'].nunique():,}")

# Set MultiIndex for linearmodels: (entity, time)
# Use permno as entity, year-quarter as time
reg_df['yyyyqq'] = (pd.PeriodIndex(
    pd.to_datetime(reg_df['date_call']).dt.to_period('Q')
).astype(str))

# Need a numeric time index
reg_df['time_idx'] = pd.Categorical(reg_df['yyyyqq']).codes
reg_df = reg_df.set_index(['permno', 'time_idx'])

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run PanelOLS with year FE + cluster at year level
# ─────────────────────────────────────────────────────────────────────────────
def run_panel_reg(df, dep_var, indep_vars, title):
    """Run panel regression with industry & year FE (dummies), SE clustered by year × industry."""
    import statsmodels.formula.api as smf

    cols_needed = ['permno', 'year', 'sic2', dep_var] + indep_vars
    sub = df[cols_needed].dropna().copy()
    sub = sub.reset_index(drop=True)

    # Add year dummies for year FE
    year_dummies = pd.get_dummies(sub['year'], prefix='yr',
                                  drop_first=True).astype(float)
    yr_cols = year_dummies.columns.tolist()

    # Add industry dummies for industry FE
    ind_dummies = pd.get_dummies(sub['sic2'], prefix='ind',
                                 drop_first=True).astype(float)
    ind_cols = ind_dummies.columns.tolist()

    sub = pd.concat([sub, year_dummies, ind_dummies], axis=1)

    # No demeaning — industry FE via dummies instead of firm FE
    all_x = indep_vars + yr_cols + ind_cols
    reg_df_ols = sub[all_x + [dep_var]].copy()
    reg_df_ols = reg_df_ols.reset_index(drop=True)

    # Two-way clustering by year × industry
    reg_df_ols['year_cluster'] = sub['year'].values
    reg_df_ols['sic2_cluster'] = sub['sic2'].values
    reg_df_ols['year_ind_cluster'] = (reg_df_ols['year_cluster'].astype(str)
                                       + '_' + reg_df_ols['sic2_cluster'].astype(str))

    formula_ols = dep_var + ' ~ ' + ' + '.join(all_x) + ' - 1'
    result = smf.ols(formula_ols, data=reg_df_ols).fit(
        cov_type='cluster',
        cov_kwds={'groups': reg_df_ols['year_ind_cluster']}
    )

    return result, indep_vars, title

def format_table(result, indep_vars, title):
    """Format regression output as a clean table."""
    coef  = result.params
    tstat = result.tvalues
    pval  = result.pvalues
    n     = int(result.nobs)
    r2    = result.rsquared

    stars = lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Dependent variable: CAR[0,1] (%)")
    print(f"  Industry & Year fixed effects: Yes")
    print(f"  SE clustered by: Year & Industry")
    print(f"{'─'*60}")
    print(f"  {'Variable':<40} {'Coef':>8}  {'t-stat':>8}  {'Sig':>4}")
    print(f"{'─'*60}")

    for var in indep_vars:
        if var in coef.index:
            c  = coef[var]
            t  = tstat[var]
            p  = pval[var]
            s  = stars(p)
            print(f"  {var:<40} {c:>8.4f}  {t:>8.3f}  {s:>4}")

    print(f"{'─'*60}")
    print(f"  Observations:  {n:,}")
    print(f"  R-squared:     {r2:.4f}")
    print(f"{'='*60}")
    print("  * p<0.10, ** p<0.05, *** p<0.01")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  REGRESSION 1: earn_surprise + ans_esg_total + pres_esg_total
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING PANEL REGRESSION 1")
print("="*60)

indep1 = []
for v in ['earn_surprise_std', 'e_talk_combined_std', 's_talk_combined_std', 'g_talk_combined_std', 'negfrac_std', 'litfrac_std' , 'uncfrac_std' , 'logword_std']:
    if v in reg_df.reset_index().columns:
        indep1.append(v)
    else:
        print(f"  WARNING: {v} not found, skipping")

if len(indep1) >= 2:
    res1, vars1, title1 = run_panel_reg(
        reg_df.reset_index(),
        'CAR_0_1',
        indep1,
        'Model 1: Total ESG Scores'
    )
    format_table(res1, vars1, title1)
else:
    print("  Not enough variables for regression 1.")
    
    
# ─────────────────────────────────────────────────────────────────────────────
# 7.a  REGRESSION 1.a: earn_surprise + ans_e_total + pres_e_total
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING PANEL REGRESSION 1.a")
print("="*60)

indep1 = []
for v in ['earn_surprise_std', 'e_talk_combined_std', 'negfrac_std', 'litfrac_std' , 'uncfrac_std' , 'logword_std']:
    if v in reg_df.reset_index().columns:
        indep1.append(v)
    else:
        print(f"  WARNING: {v} not found, skipping")

if len(indep1) >= 2:
    res1, vars1, title1 = run_panel_reg(
        reg_df.reset_index(),
        'CAR_0_1',
        indep1,
        'Model 1.a: Total E Scores'
    )
    format_table(res1, vars1, title1)
else:
    print("  Not enough variables for regression 1.")
    
    
# ─────────────────────────────────────────────────────────────────────────────
# 7.b  REGRESSION 1.b: earn_surprise + ans_s_total + pres_s_total
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING PANEL REGRESSION 1.b")
print("="*60)

indep1 = []
for v in ['earn_surprise_std', 's_talk_combined_std', 'negfrac_std', 'litfrac_std' , 'uncfrac_std' , 'logword_std']:
    if v in reg_df.reset_index().columns:
        indep1.append(v)
    else:
        print(f"  WARNING: {v} not found, skipping")

if len(indep1) >= 2:
    res1, vars1, title1 = run_panel_reg(
        reg_df.reset_index(),
        'CAR_0_1',
        indep1,
        'Model 1.b: Total S Scores'
    )
    format_table(res1, vars1, title1)
else:
    print("  Not enough variables for regression 1.b")
    
    
    
# ─────────────────────────────────────────────────────────────────────────────
# 7.c  REGRESSION 1.c: earn_surprise + ans_g_total + pres_g_total
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING PANEL REGRESSION 1.c")
print("="*60)

indep1 = []
for v in ['earn_surprise_std', 'g_talk_combined_std', 'negfrac_std', 'litfrac_std' , 'uncfrac_std' , 'logword_std']:
    if v in reg_df.reset_index().columns:
        indep1.append(v)
    else:
        print(f"  WARNING: {v} not found, skipping")

if len(indep1) >= 2:
    res1, vars1, title1 = run_panel_reg(
        reg_df.reset_index(),
        'CAR_0_1',
        indep1,
        'Model 1.c: Total G Scores'
    )
    format_table(res1, vars1, title1)
else:
    print("  Not enough variables for regression 1.c")
    
    

# ─────────────────────────────────────────────────────────────────────────────
# 8.  REGRESSION 2: earn_surprise + disaggregated ESG components
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING PANEL REGRESSION 2")
print("="*60)

indep2_candidates = [
    'earn_surprise_std',
    'e_talk_pres_std', 'e_talk_answers_std', 's_talk_pres_std', 
    's_talk_answers_std', 'g_talk_pres_std', 'g_talk_answers_std',
    'negfrac_std', 'litfrac_std' , 'uncfrac_std', 'logword_std',
]

indep2 = [v for v in indep2_candidates if v in reg_df.reset_index().columns]
missing2 = [v for v in indep2_candidates if v not in reg_df.reset_index().columns]
if missing2:
    print(f"  WARNING: missing variables: {missing2}")

if len(indep2) >= 2:
    res2, vars2, title2 = run_panel_reg(
        reg_df.reset_index(),
        'CAR_0_1',
        indep2,
        'Model 2: Disaggregated ESG Components (E, G, S)'
    )
    format_table(res2, vars2, title2)
else:
    print("  Not enough variables for regression 2.")
    
      
# ─────────────────────────────────────────────────────────────────────────────
# 9.  REGRESSION 3: earn_surprise + disaggregated E components
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING PANEL REGRESSION 3")
print("="*60)

indep3_candidates = [
    'earn_surprise_std',
    'e_talk_pres_std', 
    'e_talk_answers_std', 
    'negfrac_std', 'litfrac_std' , 'uncfrac_std', 'logword_std',
]

indep3 = [v for v in indep3_candidates if v in reg_df.reset_index().columns]
missing3 = [v for v in indep3_candidates if v not in reg_df.reset_index().columns]
if missing3:
    print(f"  WARNING: missing variables: {missing3}")

if len(indep3) >= 2:
    res3, vars3, title3 = run_panel_reg(
        reg_df.reset_index(),
        'CAR_0_1',
        indep3,
        'Model 3: Disaggregated E Components'
    )
    format_table(res3, vars3, title3)
else:
    print("  Not enough variables for regression 3.")
    
    
# ─────────────────────────────────────────────────────────────────────────────
# 10.  REGRESSION 4: earn_surprise + disaggregated S components
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING PANEL REGRESSION 4")
print("="*60)

indep4_candidates = [
    'earn_surprise_std',
    's_talk_pres_std', 
    's_talk_answers_std',
    'negfrac_std', 'litfrac_std' , 'uncfrac_std', 'logword_std',
]

indep4 = [v for v in indep4_candidates if v in reg_df.reset_index().columns]
missing4 = [v for v in indep4_candidates if v not in reg_df.reset_index().columns]
if missing4:
    print(f"  WARNING: missing variables: {missing4}")

if len(indep4) >= 2:
    res4, vars4, title4 = run_panel_reg(
        reg_df.reset_index(),
        'CAR_0_1',
        indep4,
        'Model 4: Disaggregated S Components '
    )
    format_table(res4, vars4, title4)
else:
    print("  Not enough variables for regression 4.")
        

# ─────────────────────────────────────────────────────────────────────────────
# 11.  REGRESSION 5: earn_surprise + disaggregated G components
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RUNNING PANEL REGRESSION 2")
print("="*60)

indep5_candidates = [
    'earn_surprise_std',
    'g_talk_pres_std',
    'g_talk_answers_std',
    'negfrac_std', 'litfrac_std' , 'uncfrac_std', 'logword_std',
]

indep5 = [v for v in indep5_candidates if v in reg_df.reset_index().columns]
missing5 = [v for v in indep5_candidates if v not in reg_df.reset_index().columns]
if missing5:
    print(f"  WARNING: missing variables: {missing5}")

if len(indep5) >= 2:
    res5, vars5, title5 = run_panel_reg(
        reg_df.reset_index(),
        'CAR_0_1',
        indep5,
        'Model 5: Disaggregated G Components'
    )
    format_table(res5, vars5, title5)
else:
    print("  Not enough variables for regression 5.") 
    

# ─────────────────────────────────────────────────────────────────────────────
# 12.  EXPORT COMBINED TABLE TO TXT
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_table(models, outfile="regression_results.txt"):
    """
    models: list of (result, indep_vars, model_name) tuples
    Builds a combined table with coefficients and t-stats in parentheses below.
    Saves both a plain-text (.txt) and a LaTeX (.tex) file.
    """
    # Collect all unique variables across all models (preserving order)
    all_vars = []
    for _, vars_list, _ in models:
        for v in vars_list:
            if v not in all_vars:
                all_vars.append(v)

    stars = lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))

    # Build rows: each variable gets two rows (coef row + t-stat row)
    col_width = 18
    var_width = 35

    header_names = [m[2] for m in models]
    n_models = len(models)

    lines = []

    # Header
    sep = "=" * (var_width + col_width * n_models + 2)
    lines.append(sep)
    lines.append("  Dependent variable: CAR[0,1] (%)")
    lines.append("  Industry & Year fixed effects: Yes | SE clustered by: Year & Industry")
    lines.append(sep)

    # Column headers
    header = f"  {'Variable':<{var_width}}"
    for _, _, name in models:
        short = name.replace("Model ", "M").replace(": ", " ")[:col_width-1]
        header += f"{short:>{col_width}}"
    lines.append(header)
    lines.append("-" * (var_width + col_width * n_models + 2))

    # Variable rows
    for var in all_vars:
        coef_row = f"  {var:<{var_width}}"
        tstat_row = f"  {'':<{var_width}}"

        for result, _, _ in models:
            if var in result.params.index:
                c = result.params[var]
                t = result.tvalues[var]
                p = result.pvalues[var]
                s = stars(p)
                coef_str = f"{c:.4f}{s}"
                tstat_str = f"({t:.3f})"
                coef_row  += f"{coef_str:>{col_width}}"
                tstat_row += f"{tstat_str:>{col_width}}"
            else:
                coef_row  += f"{'':>{col_width}}"
                tstat_row += f"{'':>{col_width}}"

        lines.append(coef_row)
        lines.append(tstat_row)

    # Bottom stats
    lines.append("-" * (var_width + col_width * n_models + 2))

    # Observations
    obs_row = f"  {'Observations':<{var_width}}"
    for result, _, _ in models:
        obs_row += f"{int(result.nobs):>{col_width},}"
    lines.append(obs_row)

    # R-squared
    r2_row = f"  {'R-squared':<{var_width}}"
    for result, _, _ in models:
        r2_row += f"{result.rsquared:>{col_width}.4f}"
    lines.append(r2_row)

    lines.append("=" * (var_width + col_width * n_models + 2))
    lines.append("  * p<0.10, ** p<0.05, *** p<0.01")
    lines.append("  t-statistics in parentheses")

    # Write txt file
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\nCombined table saved to: {outfile}")

    # Write LaTeX file
    tex_path = str(outfile).replace('.txt', '.tex')
    stars_fn = lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
    col_spec = 'l' + 'r' * len(models)
    tex_lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Panel Regression Results — Dependent Variable: CAR[0,1] (\%)}',
        r'\begin{tabular}{' + col_spec + '}',
        r'\hline\hline',
    ]
    # header row
    header_tex = 'Variable & ' + ' & '.join(m[2].replace('_', '\_') for m in models) + r' \\'
    tex_lines += [header_tex, r'\hline']
    for var in all_vars:
        coef_cells = []
        tstat_cells = []
        for result, _, _ in models:
            if var in result.params.index:
                c = result.params[var]
                t = result.tvalues[var]
                p = result.pvalues[var]
                s = stars_fn(p)
                coef_cells.append(f'{c:.4f}{s}')
                tstat_cells.append(f'({t:.3f})')
            else:
                coef_cells.append('')
                tstat_cells.append('')
        var_tex = var.replace('_', '\_')
        tex_lines.append(var_tex + ' & ' + ' & '.join(coef_cells) + r' \\')
        tex_lines.append(' & ' + ' & '.join(tstat_cells) + r' \\')
    tex_lines += [
        r'\hline',
        'Observations & ' + ' & '.join(f'{int(r.nobs):,}' for r, _, _ in models) + r' \\',
        'R-squared & '    + ' & '.join(f'{r.rsquared:.4f}' for r, _, _ in models) + r' \\',
        r'\hline\hline',
        r'\multicolumn{' + str(len(models)+1) + r'}{l}{\footnotesize * p$<$0.10, ** p$<$0.05, *** p$<$0.01. t-statistics in parentheses.}\\',
        r'\end{tabular}',
        r'\end{table}',
    ]
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex_lines))
    print(f'Combined table saved to: {tex_path}')


# ── Build combined table: Models 2–5 ─────────────────────────────────────────
models_2_to_5 = []
if 'res2' in dir(): models_2_to_5.append((res2, vars2, 'Model 2: All E,S,G'))
if 'res3' in dir(): models_2_to_5.append((res3, vars3, 'Model 3: E only'))
if 'res4' in dir(): models_2_to_5.append((res4, vars4, 'Model 4: S only'))
if 'res5' in dir(): models_2_to_5.append((res5, vars5, 'Model 5: G only'))

if models_2_to_5:
    build_combined_table(
        models_2_to_5,
        outfile=str(ROOT / "output" / "tables" / "regression_models2to5.txt")
    )

print("\nDone.")

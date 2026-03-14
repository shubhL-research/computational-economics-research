import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# =====================================================
# 📅 16 March 2026
# FINAL LOCKED PANEL FIXED EFFECTS MODEL
# =====================================================

# Load data
panel = pd.read_csv("final_entropy_panel.csv")

# Date formatting
panel['date'] = pd.to_datetime(panel['date'])

# Restrict to modern financial era
panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
]

print("Observations:", len(panel))

# Sort properly
panel = panel.sort_values(['country','date'])

# Create lag entropy
panel['entropy_lag1'] = panel.groupby('country')['entropy_std'].shift(1)

# Drop missing
panel = panel.dropna(subset=[
    'crisis_next_4q',
    'credit_std',
    'volatility_std',
    'entropy_std',
    'entropy_lag1'
])

print("Clean observations:", len(panel))

# =====================================================
# MODEL 1 — BASE + ENTROPY (Country FE)
# =====================================================

formula_main = """
crisis_next_4q ~ entropy_std
+ credit_std
+ volatility_std
+ C(country)
"""

model_main = smf.logit(formula_main, data=panel).fit()

print("\n===== MAIN MODEL (FE) =====")
print(model_main.summary())

# Clustered SE by country
cluster_main = model_main.get_robustcov_results(
    cov_type='cluster',
    groups=panel['country']
)

print("\n===== CLUSTERED SE (Country) =====")
print(cluster_main.summary())

# =====================================================
# MODEL 2 — Lag Robustness
# =====================================================

formula_lag = """
crisis_next_4q ~ entropy_lag1
+ credit_std
+ volatility_std
+ C(country)
"""

model_lag = smf.logit(formula_lag, data=panel).fit()

print("\n===== LAG MODEL (FE) =====")
print(model_lag.summary())

cluster_lag = model_lag.get_robustcov_results(
    cov_type='cluster',
    groups=panel['country']
)

print("\n===== LAG MODEL CLUSTERED SE =====")
print(cluster_lag.summary())

print("\n==============================")
print("📌 16 March 2026 — Empirical Backbone Locked")
print("==============================")
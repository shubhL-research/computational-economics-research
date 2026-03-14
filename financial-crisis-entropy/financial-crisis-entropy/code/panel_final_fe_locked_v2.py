import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# =====================================================
# 📅 16 March 2026
# FINAL LOCKED PANEL FE MODEL (CLUSTERED CORRECTLY)
# =====================================================

# -----------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------

panel = pd.read_csv("final_entropy_panel.csv")
panel['date'] = pd.to_datetime(panel['date'])

# Restrict to modern era
panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
]

print("Observations (modern sample):", len(panel))

# Sort panel correctly
panel = panel.sort_values(['country', 'date'])

# -----------------------------------------------------
# 2. CREATE LAG
# -----------------------------------------------------

panel['entropy_lag1'] = panel.groupby('country')['entropy_std'].shift(1)

# Drop missing values
panel = panel.dropna(subset=[
    'crisis_next_4q',
    'credit_std',
    'volatility_std',
    'entropy_std',
    'entropy_lag1'
])

print("Clean observations:", len(panel))

# -----------------------------------------------------
# 3. MAIN MODEL — COUNTRY FIXED EFFECTS
# -----------------------------------------------------

formula_main = """
crisis_next_4q ~ entropy_std
+ credit_std
+ volatility_std
+ C(country)
"""

model_main = smf.logit(
    formula_main,
    data=panel
).fit(
    cov_type='cluster',
    cov_kwds={'groups': panel['country']}
)

print("\n==============================")
print("MAIN MODEL — COUNTRY FE (CLUSTERED)")
print("==============================")
print(model_main.summary())

# -----------------------------------------------------
# 4. LAG ROBUSTNESS MODEL
# -----------------------------------------------------

formula_lag = """
crisis_next_4q ~ entropy_lag1
+ credit_std
+ volatility_std
+ C(country)
"""

model_lag = smf.logit(
    formula_lag,
    data=panel
).fit(
    cov_type='cluster',
    cov_kwds={'groups': panel['country']}
)

print("\n==============================")
print("LAG MODEL — COUNTRY FE (CLUSTERED)")
print("==============================")
print(model_lag.summary())

# -----------------------------------------------------
# 5. FINISH
# -----------------------------------------------------

print("\n==============================")
print("📌 16 March 2026 — Empirical Backbone LOCKED")
print("==============================")
import pandas as pd
import numpy as np
import statsmodels.api as sm

# =====================================================
# 📅 15 March 2026
# Entropy Persistence Test (AR1)
# =====================================================

panel = pd.read_csv("final_entropy_panel.csv")
panel['date'] = pd.to_datetime(panel['date'])

panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
].copy()

panel = panel.sort_values(['country','date'])

# Create lag
panel['entropy_lag'] = panel.groupby('country')['entropy_std'].shift(1)

panel = panel.dropna()

# ==========================================
# FULL SAMPLE AR(1)
# ==========================================

X = sm.add_constant(panel['entropy_lag'])
y = panel['entropy_std']

model = sm.OLS(y, X).fit()

print("\n=== FULL SAMPLE AR(1) ===")
print(model.summary())

# ==========================================
# POST-CRISIS SAMPLE ONLY
# ==========================================

post = panel[panel['crisis_next_4q'] == 1]

X_post = sm.add_constant(post['entropy_lag'])
y_post = post['entropy_std']

model_post = sm.OLS(y_post, X_post).fit()

print("\n=== CRISIS PERIOD AR(1) ===")
print(model_post.summary())

# ==========================================
# NORMAL PERIOD SAMPLE
# ==========================================

normal = panel[panel['crisis_next_4q'] == 0]

X_norm = sm.add_constant(normal['entropy_lag'])
y_norm = normal['entropy_std']

model_norm = sm.OLS(y_norm, X_norm).fit()

print("\n=== NORMAL PERIOD AR(1) ===")
print(model_norm.summary())
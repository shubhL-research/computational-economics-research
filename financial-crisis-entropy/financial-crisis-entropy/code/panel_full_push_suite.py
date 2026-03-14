import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

# =====================================================
# 📅 12 March 2026 — FULL PUSH SUITE
# Modern sample + GFC validation window
# =====================================================

# Load panel
panel = pd.read_csv("final_entropy_panel.csv")
panel['date'] = pd.to_datetime(panel['date'])

# Modern era only
panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
].copy()

print("Modern sample observations:", len(panel))

# Create 1Q lag entropy
panel = panel.sort_values(['country','date'])
panel['entropy_lag1'] = panel.groupby('country')['entropy_std'].shift(1)

# Interaction term
panel['interaction'] = panel['entropy_std'] * panel['credit_std']

# Drop missing
panel = panel.dropna()

# Time split
train = panel[panel['date'] < '2007-01-01']
test  = panel[(panel['date'] >= '2007-01-01') &
              (panel['date'] <= '2014-12-31')]

print("Train size:", len(train))
print("Test size:", len(test))

y_train = train['crisis_next_4q']
y_test  = test['crisis_next_4q']

# =====================================================
# FUNCTION TO RUN MODEL
# =====================================================

def run_model(features, name):

    X_train = sm.add_constant(train[features])
    X_test  = sm.add_constant(test[features])

    model = sm.Logit(y_train, X_train).fit(disp=False)

    auc_train = roc_auc_score(y_train, model.predict(X_train))
    auc_test  = roc_auc_score(y_test, model.predict(X_test))

    print("\n======", name, "======")
    print("Train AUC:", round(auc_train,4))
    print("Test AUC :", round(auc_test,4))

# =====================================================
# RUN ALL MODELS
# =====================================================

# 1. Base
run_model(['credit_std','volatility_std'], "BASE")

# 2. + Entropy
run_model(['credit_std','volatility_std','entropy_std'], "+ENTROPY")

# 3. Interaction
run_model(['credit_std','volatility_std','entropy_std','interaction'], "INTERACTION")

# 4. Lag entropy
run_model(['credit_std','volatility_std','entropy_lag1'], "LAG 1Q ENTROPY")
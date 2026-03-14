import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

# =====================================================
# 📅 PUSH MODEL — 12 March 2026
# Modern Sample + Proper Time Split
# =====================================================

# Load panel
panel = pd.read_csv("final_entropy_panel.csv")
panel['date'] = pd.to_datetime(panel['date'])

# Restrict to modern financial era
panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
]

print("Modern sample observations:", len(panel))

# =====================================================
# Create time-based split (NO random split)
# Train: Before 2015
# Test: 2015 and after
# =====================================================

train = panel[panel['date'] < '2007-01-01']
test  = panel[(panel['date'] >= '2007-01-01') & (panel['date'] <= '2014-12-31')]
print("Train size:", len(train))
print("Test size:", len(test))

# =====================================================
# BASE MODEL (Credit + Volatility)
# =====================================================

X_train_base = train[['credit_std','volatility_std']]
X_train_base = sm.add_constant(X_train_base)

y_train = train['crisis_next_4q']

model_base = sm.Logit(y_train, X_train_base).fit(disp=False)

# In-sample AUC
train_pred_base = model_base.predict(X_train_base)
auc_train_base = roc_auc_score(y_train, train_pred_base)

# Out-of-sample AUC
X_test_base = test[['credit_std','volatility_std']]
X_test_base = sm.add_constant(X_test_base)

y_test = test['crisis_next_4q']
test_pred_base = model_base.predict(X_test_base)
auc_test_base = roc_auc_score(y_test, test_pred_base)

# =====================================================
# + ENTROPY MODEL
# =====================================================

X_train_ent = train[['credit_std','volatility_std','entropy_std']]
X_train_ent = sm.add_constant(X_train_ent)

model_ent = sm.Logit(y_train, X_train_ent).fit(disp=False)

train_pred_ent = model_ent.predict(X_train_ent)
auc_train_ent = roc_auc_score(y_train, train_pred_ent)

X_test_ent = test[['credit_std','volatility_std','entropy_std']]
X_test_ent = sm.add_constant(X_test_ent)

test_pred_ent = model_ent.predict(X_test_ent)
auc_test_ent = roc_auc_score(y_test, test_pred_ent)

# =====================================================
# RESULTS
# =====================================================

print("\n==============================")
print("IN-SAMPLE AUC")
print("Base:", round(auc_train_base,4))
print("+Entropy:", round(auc_train_ent,4))

print("\nOUT-OF-SAMPLE AUC")
print("Base:", round(auc_test_base,4))
print("+Entropy:", round(auc_test_ent,4))
print("Difference:", round(auc_test_ent - auc_test_base,4))
print("==============================")
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# =========================================
# 📅 Research Freeze Date: 12 March 2026
# Final Model Specification
# =========================================

# LOAD FINAL MERGED PANEL
panel = pd.read_csv("final_entropy_panel.csv")

panel['date'] = pd.to_datetime(panel['date'])

# Drop missing
panel = panel.dropna(subset=[
    'credit_std',
    'volatility_std',
    'entropy_std',
    'crisis_next_4q'
])

print("Total observations:", len(panel))

# =========================================
# FIXED EFFECTS
# =========================================

dummies = pd.get_dummies(panel['country'], drop_first=True)

X_base = pd.concat([
    panel[['credit_std','volatility_std']],
    dummies
], axis=1)

X_full = pd.concat([
    panel[['credit_std','volatility_std','entropy_std']],
    dummies
], axis=1)

X_base = sm.add_constant(X_base).astype(float)
X_full = sm.add_constant(X_full).astype(float)

y = panel['crisis_next_4q'].astype(int)

# =========================================
# TRAIN TEST SPLIT (Time-based)
# =========================================

split_date = panel['date'].quantile(0.6)

train = panel[panel['date'] <= split_date]
test  = panel[panel['date'] > split_date]

# Rebuild X matrices for train/test
Xb_train = sm.add_constant(pd.concat([
    train[['credit_std','volatility_std']],
    pd.get_dummies(train['country'], drop_first=True)
], axis=1)).astype(float)

Xf_train = sm.add_constant(pd.concat([
    train[['credit_std','volatility_std','entropy_std']],
    pd.get_dummies(train['country'], drop_first=True)
], axis=1)).astype(float)

Xb_test = sm.add_constant(pd.concat([
    test[['credit_std','volatility_std']],
    pd.get_dummies(test['country'], drop_first=True)
], axis=1)).astype(float)

Xf_test = sm.add_constant(pd.concat([
    test[['credit_std','volatility_std','entropy_std']],
    pd.get_dummies(test['country'], drop_first=True)
], axis=1)).astype(float)

y_train = train['crisis_next_4q'].astype(int)
y_test  = test['crisis_next_4q'].astype(int)

# Align columns
Xb_test = Xb_test.reindex(columns=Xb_train.columns, fill_value=0)
Xf_test = Xf_test.reindex(columns=Xf_train.columns, fill_value=0)

# =========================================
# FIT MODELS
# =========================================

model_base = sm.Logit(y_train, Xb_train)
result_base = model_base.fit(disp=0)

model_full = sm.Logit(y_train, Xf_train)
result_full = model_full.fit(disp=0)

print("\n===== BASELINE MODEL =====")
print(result_base.summary())

print("\n===== + ENTROPY MODEL =====")
print(result_full.summary())

# =========================================
# AUC
# =========================================

pred_base_train = result_base.predict(Xb_train)
pred_full_train = result_full.predict(Xf_train)

pred_base_test = result_base.predict(Xb_test)
pred_full_test = result_full.predict(Xf_test)

print("\n==============================")
print("In-sample AUC (Base):", round(roc_auc_score(y_train, pred_base_train),4))
print("In-sample AUC (+Entropy):", round(roc_auc_score(y_train, pred_full_train),4))
print("Difference:", round(
    roc_auc_score(y_train, pred_full_train)
    - roc_auc_score(y_train, pred_base_train),4))

print("\nOut-of-sample AUC (Base):", round(roc_auc_score(y_test, pred_base_test),4))
print("Out-of-sample AUC (+Entropy):", round(roc_auc_score(y_test, pred_full_test),4))
print("Difference:", round(
    roc_auc_score(y_test, pred_full_test)
    - roc_auc_score(y_test, pred_base_test),4))
print("==============================\n")
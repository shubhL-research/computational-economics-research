import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from sklearn.metrics import roc_auc_score

# =========================================
# 📅 Research Date: 3 March 2026
# AUC Comparison: Credit vs Credit+Entropy
# =========================================

countries = ['US','UK','DE','FR','IT','ES','NL','SE','CA','JP']

# =========================================
# LOAD BASE DATA
# =========================================

data_list = []

for c in countries:
    path = f"{c}_RESULTS/{c.lower()}_entropy_full_dataset.csv"
    
    if os.path.exists(path):
        df = pd.read_csv(path)

        if 'date' not in df.columns and 'observation_date' in df.columns:
            df = df.rename(columns={'observation_date':'date'})

        df['country'] = c
        data_list.append(df)
    else:
        print(f"Missing entropy file for {c}")

panel = pd.concat(data_list, ignore_index=True)

panel['date'] = pd.to_datetime(panel['date'], errors='coerce')
panel = panel.dropna(subset=['date'])
panel = panel.sort_values(['country','date'])

# =========================================
# LOAD CREDIT
# =========================================

credit_list = []

for c in countries:
    file = f"{c.lower()}_credit.csv"
    
    if os.path.exists(file):
        df = pd.read_csv(file)

        if 'observation_date' in df.columns:
            df = df.rename(columns={'observation_date':'date'})

        credit_col = df.columns[1]
        df = df.rename(columns={credit_col:'credit'})

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')

        df['credit_yoy'] = df['credit'].pct_change(4)
        df['country'] = c

        credit_list.append(df[['date','country','credit_yoy']])
    else:
        print(f"Missing credit file for {c}")

credit_panel = pd.concat(credit_list, ignore_index=True)

# =========================================
# MERGE
# =========================================

panel = panel.merge(
    credit_panel,
    on=['date','country'],
    how='left'
)

panel = panel.dropna(subset=[
    'volatility_std',
    'credit_yoy',
    'entropy_std',
    'crisis_next_4q'
])

print("Total observations:", len(panel))

# =========================================
# STANDARDIZE VARIABLES
# =========================================

panel['credit_std'] = (
    (panel['credit_yoy'] - panel['credit_yoy'].mean())
    / panel['credit_yoy'].std()
)

panel['entropy_std2'] = (
    (panel['entropy_std'] - panel['entropy_std'].mean())
    / panel['entropy_std'].std()
)

y = panel['crisis_next_4q'].astype(int)

dummies = pd.get_dummies(panel['country'], drop_first=True)

# =========================================
# MODEL A: Credit + Volatility
# =========================================

X_A = pd.concat([
    panel[['credit_std','volatility_std']],
    dummies
], axis=1)

X_A = sm.add_constant(X_A).astype(float)

model_A = sm.Logit(y, X_A).fit(disp=False)
panel['pred_A'] = model_A.predict(X_A)

auc_A = roc_auc_score(y, panel['pred_A'])

# =========================================
# MODEL B: Credit + Volatility + Entropy
# =========================================

X_B = pd.concat([
    panel[['credit_std','volatility_std','entropy_std2']],
    dummies
], axis=1)

X_B = sm.add_constant(X_B).astype(float)

model_B = sm.Logit(y, X_B).fit(disp=False)
panel['pred_B'] = model_B.predict(X_B)

auc_B = roc_auc_score(y, panel['pred_B'])

# =========================================
# PRINT RESULTS
# =========================================

print("\n=============================")
print("AUC Model A (Credit only):", round(auc_A,4))
print("AUC Model B (+ Entropy):  ", round(auc_B,4))
print("Difference:", round(auc_B - auc_A,4))
print("=============================\n")
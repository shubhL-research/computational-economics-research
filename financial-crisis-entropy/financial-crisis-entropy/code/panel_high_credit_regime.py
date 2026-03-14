import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from sklearn.metrics import roc_auc_score

# =========================================
# 📅 Research Date: 5 March 2026
# High Credit Regime Model
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

# =========================================
# HIGH CREDIT REGIME (Top 25%)
# =========================================

threshold = panel['credit_std'].quantile(0.75)

panel['high_credit'] = (panel['credit_std'] > threshold).astype(int)

# Interaction term
panel['entropy_high_credit'] = panel['entropy_std2'] * panel['high_credit']

# =========================================
# FIXED EFFECTS
# =========================================

dummies = pd.get_dummies(panel['country'], drop_first=True)

X = pd.concat([
    panel[['credit_std',
           'volatility_std',
           'entropy_std2',
           'entropy_high_credit']],
    dummies
], axis=1)

X = sm.add_constant(X).astype(float)

y = panel['crisis_next_4q'].astype(int)

# =========================================
# LOGIT MODEL
# =========================================

model = sm.Logit(y, X)
result = model.fit(
    cov_type='cluster',
    cov_kwds={'groups': panel['country']}
)

print(result.summary())

# =========================================
# AUC
# =========================================

panel['pred_prob'] = result.predict(X)
auc = roc_auc_score(y, panel['pred_prob'])

print("\n=============================")
print("AUC with High Credit Regime:", round(auc,4))
print("=============================\n")
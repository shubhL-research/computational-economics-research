import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# =========================================
# SETTINGS
# =========================================

countries = ['US','UK','DE','FR','IT','ES','NL','SE','CA','JP']

# =========================================
# LOAD ENTROPY FILES
# =========================================

entropy_list = []

for c in countries:
    path = f"{c}_RESULTS/{c.lower()}_entropy_full_dataset.csv"
    
    if os.path.exists(path):
        df = pd.read_csv(path)

        # Ensure correct date column
        if 'date' not in df.columns:
            if 'observation_date' in df.columns:
                df = df.rename(columns={'observation_date':'date'})
        
        df['country'] = c
        entropy_list.append(df)
    else:
        print(f"Missing entropy file for {c}")

panel = pd.concat(entropy_list, ignore_index=True)

# Fix dates properly
panel['date'] = pd.to_datetime(panel['date'], errors='coerce')
panel = panel.dropna(subset=['date'])
panel = panel.sort_values(['country','date'])

# =========================================
# CREATE DELTA ENTROPY
# =========================================

panel['delta_entropy'] = (
    panel.groupby('country')['entropy_std'].diff()
)

# 1Q lag of delta entropy
panel['delta_entropy_lag1'] = (
    panel.groupby('country')['delta_entropy'].shift(1)
)

# =========================================
# LOAD CREDIT FILES
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
    'delta_entropy_lag1',
    'volatility_std',
    'credit_yoy',
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

panel['delta_entropy_std'] = (
    (panel['delta_entropy_lag1'] - panel['delta_entropy_lag1'].mean())
    / panel['delta_entropy_lag1'].std()
)

# Interaction
panel['entropy_credit_interaction'] = (
    panel['delta_entropy_std'] * panel['credit_std']
)

# =========================================
# FIXED EFFECTS
# =========================================

dummies = pd.get_dummies(panel['country'], drop_first=True)

X = pd.concat([
    panel[['delta_entropy_std',
           'volatility_std',
           'credit_std',
           'entropy_credit_interaction']],
    dummies
], axis=1)

X = sm.add_constant(X)
X = X.astype(float)

y = panel['crisis_next_4q'].astype(int)

# =========================================
# LOGIT WITH CLUSTERED SE
# =========================================

model = sm.Logit(y, X)

result = model.fit(
    cov_type='cluster',
    cov_kwds={'groups': panel['country']}
)

print(result.summary())
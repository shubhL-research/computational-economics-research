import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# =========================================
# 1. SETTINGS
# =========================================

countries = ['US','UK','DE','FR','IT','ES','NL','SE','CA','JP']

# =========================================
# 2. LOAD ENTROPY COUNTRY FILES
# =========================================

entropy_list = []

for c in countries:
    path = f"{c}_RESULTS/{c.lower()}_entropy_full_dataset.csv"
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['country'] = c
        entropy_list.append(df)
    else:
        print(f"Missing entropy file for {c}")

panel = pd.concat(entropy_list, ignore_index=True)

# Ensure date format
panel['date'] = pd.to_datetime(panel['date'])

# =========================================
# 3. LOAD CREDIT FILES
# =========================================

credit_list = []

for c in countries:
    file = f"{c.lower()}_credit.csv"
    
    if os.path.exists(file):
        df = pd.read_csv(file)

        # second column is credit series
        credit_col = df.columns[1]

        df = df.rename(columns={
            'observation_date': 'date',
            credit_col: 'credit'
        })

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # Year-over-year growth
        df['credit_yoy'] = df['credit'].pct_change(4)

        df['country'] = c

        credit_list.append(df[['date','country','credit_yoy']])
    else:
        print(f"Missing credit file for {c}")

credit_panel = pd.concat(credit_list, ignore_index=True)

# =========================================
# 4. MERGE ENTROPY + CREDIT
# =========================================

panel = panel.merge(
    credit_panel,
    on=['date','country'],
    how='left'
)

# Drop rows with missing required vars
panel = panel.dropna(subset=[
    'entropy_std',
    'volatility_std',
    'credit_yoy',
    'crisis_next_4q'
])

print("Total observations after merge:", len(panel))

# =========================================
# 5. STANDARDIZE CREDIT
# =========================================

panel['credit_std'] = (
    (panel['credit_yoy'] - panel['credit_yoy'].mean())
    / panel['credit_yoy'].std()
)

# =========================================
# 6. BUILD FIXED EFFECTS
# =========================================

dummies = pd.get_dummies(panel['country'], drop_first=True)

X = pd.concat([
    panel[['entropy_std','volatility_std','credit_std']],
    dummies
], axis=1)

X = sm.add_constant(X)

# Force numeric
X = X.astype(float)

y = panel['crisis_next_4q'].astype(int)

# =========================================
# 7. LOGIT WITH CLUSTERED SE
# =========================================

model = sm.Logit(y, X)

result = model.fit(
    cov_type='cluster',
    cov_kwds={'groups': panel['country']}
)

print(result.summary())
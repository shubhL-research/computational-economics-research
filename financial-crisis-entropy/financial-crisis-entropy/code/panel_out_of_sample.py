import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from sklearn.metrics import roc_auc_score

# =========================================
# 📅 Research Date: 4 March 2026
# Out-of-Sample Evaluation
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

# =========================================
# STANDARDIZE
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
# TRAIN / TEST SPLIT
# =========================================

train = panel[panel['date'] < '2008-01-01']
test  = panel[panel['date'] >= '2008-01-01']

print("Train size:", len(train))
print("Test size:", len(test))

y_train = train['crisis_next_4q'].astype(int)
y_test  = test['crisis_next_4q'].astype(int)

dummies_train = pd.get_dummies(train['country'], drop_first=True)
dummies_test  = pd.get_dummies(test['country'], drop_first=True)

# Align dummy columns
dummies_test = dummies_test.reindex(columns=dummies_train.columns, fill_value=0)

# =========================================
# MODEL A (Credit Only)
# =========================================

X_train_A = pd.concat([
    train[['credit_std','volatility_std']],
    dummies_train
], axis=1)

X_test_A = pd.concat([
    test[['credit_std','volatility_std']],
    dummies_test
], axis=1)

X_train_A = sm.add_constant(X_train_A).astype(float)
X_test_A  = sm.add_constant(X_test_A).astype(float)

model_A = sm.Logit(y_train, X_train_A).fit(disp=False)

pred_test_A = model_A.predict(X_test_A)
auc_A = roc_auc_score(y_test, pred_test_A)

# =========================================
# MODEL B (+ Entropy)
# =========================================

X_train_B = pd.concat([
    train[['credit_std','volatility_std','entropy_std2']],
    dummies_train
], axis=1)

X_test_B = pd.concat([
    test[['credit_std','volatility_std','entropy_std2']],
    dummies_test
], axis=1)

X_train_B = sm.add_constant(X_train_B).astype(float)
X_test_B  = sm.add_constant(X_test_B).astype(float)

model_B = sm.Logit(y_train, X_train_B).fit(disp=False)

pred_test_B = model_B.predict(X_test_B)
auc_B = roc_auc_score(y_test, pred_test_B)

# =========================================
# RESULTS
# =========================================

print("\n==============================")
print("Out-of-Sample AUC (Credit):", round(auc_A,4))
print("Out-of-Sample AUC (+Entropy):", round(auc_B,4))
print("Difference:", round(auc_B - auc_A,4))
print("==============================\n")
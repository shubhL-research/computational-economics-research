import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from sklearn.metrics import roc_auc_score

# =========================================
# 📅 Research Date: 8 March 2026
# Systemic Credit Entropy Model
# =========================================

countries = ['US','UK','DE','FR','ES','NL','SE','CA','JP']

# =========================================
# LOAD CREDIT DATA
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
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        df['credit_yoy'] = df['credit'].pct_change(4)
        df['country'] = c
        
        credit_list.append(df[['date','country','credit_yoy']])
    else:
        print(f"Missing credit file for {c}")

credit_panel = pd.concat(credit_list, ignore_index=True)

# =========================================
# BUILD SYSTEM ENTROPY
# =========================================

entropy_list = []

for date, group in credit_panel.groupby('date'):
    
    g = group.dropna(subset=['credit_yoy'])
    
    if len(g) < 5:
        continue
    
    values = np.abs(g['credit_yoy'].values)
    
    total = np.sum(values)
    
    if total == 0:
        continue
    
    probs = values / total
    
    entropy = -np.sum(probs * np.log(probs))
    
    entropy_list.append({
        'date': date,
        'system_entropy': entropy
    })

system_entropy_df = pd.DataFrame(entropy_list)

# =========================================
# LOAD ENTROPY PANEL
# =========================================

panel_list = []

for c in countries:
    path = f"{c}_RESULTS/{c.lower()}_entropy_full_dataset.csv"
    
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        if 'observation_date' in df.columns:
            df = df.rename(columns={'observation_date':'date'})
        
        df['date'] = pd.to_datetime(df['date'])
        df['country'] = c
        
        panel_list.append(df)
    else:
        print(f"Missing entropy file for {c}")

panel = pd.concat(panel_list, ignore_index=True)

# =========================================
# MERGE SYSTEM ENTROPY
# =========================================

panel = panel.merge(system_entropy_df, on='date', how='left')
panel = panel.merge(credit_panel, on=['date','country'], how='left')

panel = panel.dropna(subset=[
    'system_entropy',
    'credit_yoy',
    'volatility_std',
    'crisis_next_4q'
])

print("Total observations:", len(panel))

# =========================================
# STANDARDIZE
# =========================================

panel['credit_std'] = (panel['credit_yoy'] - panel['credit_yoy'].mean()) / panel['credit_yoy'].std()
panel['system_entropy_std'] = (panel['system_entropy'] - panel['system_entropy'].mean()) / panel['system_entropy'].std()

# =========================================
# FIXED EFFECTS
# =========================================

dummies = pd.get_dummies(panel['country'], drop_first=True)

X = pd.concat([
    panel[['credit_std','volatility_std','system_entropy_std']],
    dummies
], axis=1)

X = sm.add_constant(X).astype(float)
y = panel['crisis_next_4q'].astype(int)

# =========================================
# MODEL
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
print("System Entropy Model AUC:", round(auc,4))
print("=============================\n")
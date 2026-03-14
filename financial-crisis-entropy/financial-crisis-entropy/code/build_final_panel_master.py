import pandas as pd
import numpy as np
import os

# =====================================================
# 📅 Final Panel Build Date: 12 March 2026
# MASTER VERSION — Handles IT special filename
# =====================================================

countries = ['US','UK','DE','FR','IT','ES','NL','SE','CA','JP']

panel_list = []

for c in countries:

    # ---- Handle Italy special filename ----
    if c == "IT":
        entropy_path = "IT_RESULTS/it_full_dataset.csv"
    else:
        entropy_path = f"{c}_RESULTS/{c.lower()}_entropy_full_dataset.csv"

    credit_path = f"{c.lower()}_credit.csv"

    if not os.path.exists(entropy_path):
        print(f"Missing entropy file for {c}")
        continue

    if not os.path.exists(credit_path):
        print(f"Missing credit file for {c}")
        continue

    # -------------------------
    # LOAD ENTROPY DATA
    # -------------------------
    entropy_df = pd.read_csv(entropy_path)

    # Standardize column names
    if 'observation_date' in entropy_df.columns:
        entropy_df = entropy_df.rename(columns={'observation_date':'date'})

    entropy_df['date'] = pd.to_datetime(entropy_df['date'])
    entropy_df['country'] = c

    # -------------------------
    # LOAD CREDIT DATA
    # -------------------------
    credit_df = pd.read_csv(credit_path)

    if 'observation_date' in credit_df.columns:
        credit_df = credit_df.rename(columns={'observation_date':'date'})

    credit_col = credit_df.columns[1]
    credit_df = credit_df.rename(columns={credit_col:'credit'})

    credit_df['date'] = pd.to_datetime(credit_df['date'])
    credit_df = credit_df.sort_values('date')

    # YoY credit growth
    credit_df['credit_yoy'] = credit_df['credit'].pct_change(4)
    credit_df['country'] = c

    # -------------------------
    # MERGE
    # -------------------------
    merged = entropy_df.merge(
        credit_df[['date','country','credit_yoy']],
        on=['date','country'],
        how='left'
    )

    panel_list.append(merged)

# Combine all countries
panel = pd.concat(panel_list, ignore_index=True)

# Drop missing key variables
panel = panel.dropna(subset=[
    'credit_yoy',
    'volatility_std',
    'entropy_std',
    'crisis_next_4q'
])

print("Final merged observations:", len(panel))

# -------------------------
# STANDARDIZE VARIABLES
# -------------------------
panel['credit_std'] = (
    panel['credit_yoy'] - panel['credit_yoy'].mean()
) / panel['credit_yoy'].std()

panel['entropy_std'] = (
    panel['entropy_std'] - panel['entropy_std'].mean()
) / panel['entropy_std'].std()

panel['volatility_std'] = (
    panel['volatility_std'] - panel['volatility_std'].mean()
) / panel['volatility_std'].std()

# Save final panel
panel.to_csv("final_entropy_panel.csv", index=False)

print("Saved as final_entropy_panel.csv")
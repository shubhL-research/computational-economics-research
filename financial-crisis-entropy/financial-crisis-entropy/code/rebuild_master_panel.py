import pandas as pd

# ======================================================
# MASTER PANEL REBUILD — 19 March 2026
# ======================================================

# Load entropy
entropy = pd.read_csv("entropy_panel_clean.csv")
entropy['date'] = pd.to_datetime(entropy['date'])

# Load crisis dataset
crisis = pd.read_csv("Crisis Dataset.csv")
crisis['date'] = pd.to_datetime(crisis['date'])

# Load credit master (already standardized one)
credit = pd.read_csv("panel_credit_master.csv")
credit['date'] = pd.to_datetime(credit['date'])

# Load volatility if separate
vol = pd.read_csv("panel_volatility.csv")
vol['date'] = pd.to_datetime(vol['date'])

# Merge step by step
panel = entropy.merge(credit, on=["date","country"], how="inner")
panel = panel.merge(vol, on=["date","country"], how="inner")
panel = panel.merge(crisis, on=["date","country"], how="inner")

panel = panel.dropna()

panel.to_csv("FINAL_PANEL_CLEAN.csv", index=False)

print("FINAL PANEL CREATED")
print("Total observations:", len(panel))
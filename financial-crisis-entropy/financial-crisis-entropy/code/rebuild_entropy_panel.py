import pandas as pd
import numpy as np
import glob
import os

# ======================================================
# CLEAN ENTROPY RECONSTRUCTION — 18 March 2026
# ======================================================

WINDOW = 8
BINS = 10

def shannon_entropy(series):
    hist, _ = np.histogram(series, bins=BINS, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log(hist))

def rolling_entropy(series):
    return series.rolling(WINDOW).apply(shannon_entropy, raw=False)

# ------------------------------------------------------
# 1. Process each country credit file
# ------------------------------------------------------

credit_files = glob.glob("*_credit.csv")

entropy_frames = []

for file in credit_files:
    
    country = file.split("_")[0].upper()
    df = pd.read_csv(file)
    
    # Standardize date column
    if "observation_date" in df.columns:
        df = df.rename(columns={"observation_date":"date"})
    
    df['date'] = pd.to_datetime(df['date'])
    
    # Assume second column is credit series
    credit_col = df.columns[1]
    
    df = df.sort_values("date")
    
    df['entropy_raw'] = rolling_entropy(df[credit_col])
    
    # Standardize within country
    df['entropy_std'] = (
        df['entropy_raw'] - df['entropy_raw'].mean()
    ) / df['entropy_raw'].std()
    
    df = df[['date','entropy_std']]
    df['country'] = country
    
    entropy_frames.append(df)

# ------------------------------------------------------
# 2. Combine all countries
# ------------------------------------------------------

entropy_panel = pd.concat(entropy_frames)
entropy_panel = entropy_panel.dropna()

entropy_panel.to_csv("entropy_panel_clean.csv", index=False)

print("Entropy panel created successfully.")
print("Total observations:", len(entropy_panel))
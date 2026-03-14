import pandas as pd
import numpy as np

# ==========================================
# CLEAN ENTROPY CONSTRUCTION (REPRODUCIBLE)
# ==========================================

WINDOW = 8      # 8 quarters
BINS = 10       # histogram bins

def compute_entropy(series):
    hist, _ = np.histogram(series, bins=BINS, density=True)
    hist = hist[hist > 0]  # remove zeros
    return -np.sum(hist * np.log(hist))

def rolling_entropy(series):
    return series.rolling(WINDOW).apply(compute_entropy, raw=False)

# Example usage:
df = pd.read_csv("us_credit.csv")
df['entropy'] = rolling_entropy(df['credit_growth'])

# Standardize
df['entropy_std'] = (
    df['entropy'] - df['entropy'].mean()
) / df['entropy'].std()

df.to_csv("us_entropy_clean.csv", index=False)
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

# ===============================
# LOAD DATA
# ===============================

df = pd.read_csv("au_gdp.csv")

# Rename columns safely
df.rename(columns={
    "observation_date": "date",
    df.columns[1]: "gdp"
}, inplace=True)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ===============================
# GDP GROWTH (YoY)
# ===============================

df["gdp_growth"] = df["gdp"].pct_change(4)

# ===============================
# ROLLING ENTROPY
# ===============================

window = 12

def rolling_entropy(series, window):
    entropy_values = []

    for i in range(len(series)):
        if i < window:
            entropy_values.append(np.nan)
        else:
            segment = series[i-window:i]

            # Shift to positive
            segment = segment - np.min(segment) + 1e-6

            probs = segment / np.sum(segment)
            entropy = -np.sum(probs * np.log(probs))
            entropy_values.append(entropy)

    return entropy_values

df["entropy"] = rolling_entropy(df["gdp_growth"].values, window)

# ===============================
# VOLATILITY
# ===============================

df["volatility"] = df["gdp_growth"].rolling(window).std()

# ===============================
# CRISIS VARIABLE
# Australia did not collapse in 2008
# Only mark COVID shock
# ===============================

df["year"] = df["date"].dt.year

crisis_years = [2020]

df["crisis"] = df["year"].isin(crisis_years).astype(int)

# 4-quarter forward crisis
df["crisis_next_4q"] = df["crisis"].shift(-4)

# ===============================
# CLEAN
# ===============================

df = df.dropna()

# ===============================
# STANDARDIZE
# ===============================

df["entropy_std"] = (df["entropy"] - df["entropy"].mean()) / df["entropy"].std()
df["volatility_std"] = (df["volatility"] - df["volatility"].mean()) / df["volatility"].std()

# ===============================
# LOGISTIC REGRESSION
# ===============================

X = df[["entropy_std", "volatility_std"]]
y = df["crisis_next_4q"]

model = LogisticRegression()
model.fit(X, y)

print("\nAUSTRALIA RESULTS")
print("Entropy coefficient:", model.coef_[0][0])
print("Volatility coefficient:", model.coef_[0][1])

# ===============================
# SAVE DATASET
# ===============================

os.makedirs("AU_RESULTS", exist_ok=True)
df.to_csv("AU_RESULTS/au_entropy_full_dataset.csv", index=False)

print("AU_RESULTS folder fully saved.")
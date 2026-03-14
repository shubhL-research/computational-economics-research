import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression

# ---------- LOAD ----------
df = pd.read_csv("se_gdp.csv")

df.rename(columns={
    "observation_date": "date",
    df.columns[1]: "gdp"
}, inplace=True)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ---------- GDP GROWTH ----------
df["gdp_growth"] = df["gdp"].pct_change(4)

# ---------- ENTROPY ----------
window = 12

def rolling_entropy(series):
    ent = []
    for i in range(len(series)):
        if i < window:
            ent.append(np.nan)
        else:
            s = series[i-window:i]
            s = s - np.min(s) + 1e-6
            p = s / np.sum(s)
            ent.append(-np.sum(p * np.log(p)))
    return ent

df["entropy"] = rolling_entropy(df["gdp_growth"].values)

# ---------- VOLATILITY ----------
df["volatility"] = df["gdp_growth"].rolling(window).std()

# ---------- CRISIS ----------
df["year"] = df["date"].dt.year
crisis_years = [2008]
df["crisis"] = df["year"].isin(crisis_years).astype(int)

# ---------- FORWARD ----------
df["crisis_next_4q"] = df["crisis"].shift(-4)

# ---------- CLEAN ----------
df = df.dropna()

# ---------- STANDARDIZE ----------
df["entropy_std"] = (df["entropy"] - df["entropy"].mean()) / df["entropy"].std()
df["volatility_std"] = (df["volatility"] - df["volatility"].mean()) / df["volatility"].std()

# ---------- LOGISTIC ----------
X = df[["entropy_std", "volatility_std"]]
y = df["crisis_next_4q"]

model = LogisticRegression()
model.fit(X, y)

print("SWEDEN RESULTS")
print("Entropy coefficient:", model.coef_[0][0])
print("Volatility coefficient:", model.coef_[0][1])

# ---------- SAVE ----------
os.makedirs("SE_RESULTS", exist_ok=True)
df.to_csv("SE_RESULTS/se_entropy_full_dataset.csv", index=False)

print("SE_RESULTS folder fully saved.")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# ---------------------------------------
# CREATE FR_RESULTS FOLDER
# ---------------------------------------
folder_path = "FR_RESULTS"
os.makedirs(folder_path, exist_ok=True)

# ---------------------------------------
# LOAD FRANCE GDP
# ---------------------------------------
df = pd.read_csv("fr_gdp.csv")

df.rename(columns={
    "observation_date": "date",
    "CLVMNACSCAB1GQFR": "gdp"
}, inplace=True)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------------------------------------
# GDP GROWTH (YoY)
# ---------------------------------------
df["gdp_growth"] = df["gdp"].pct_change(4)
df = df.dropna().reset_index(drop=True)

# ---------------------------------------
# ENTROPY + VOLATILITY (WINDOW = 12)
# ---------------------------------------
window = 12
df["entropy"] = np.nan
df["volatility"] = np.nan

for i in range(window, len(df)):
    window_data = df["gdp_growth"].iloc[i-window:i]

    p = np.abs(window_data)
    p = p / np.sum(p)
    H = -np.sum(p * np.log(p))
    df.loc[i, "entropy"] = H

    df.loc[i, "volatility"] = np.std(window_data)

# ---------------------------------------
# CRISIS DUMMY (France 2008)
# ---------------------------------------
df["year"] = df["date"].dt.year
df["crisis"] = 0
df.loc[df["year"] == 2008, "crisis"] = 1

# ---------------------------------------
# FORWARD 4Q CRISIS
# ---------------------------------------
df["crisis_next_4q"] = 0

for i in range(len(df)-4):
    if df.loc[i+1:i+4, "crisis"].sum() > 0:
        df.loc[i, "crisis_next_4q"] = 1

# ---------------------------------------
# STANDARDIZE VARIABLES
# ---------------------------------------
model_df = df.dropna(subset=["entropy", "volatility"]).copy()

model_df["entropy_std"] = (
    model_df["entropy"] - model_df["entropy"].mean()
) / model_df["entropy"].std()

model_df["volatility_std"] = (
    model_df["volatility"] - model_df["volatility"].mean()
) / model_df["volatility"].std()

# ---------------------------------------
# LOGISTIC REGRESSION
# ---------------------------------------
X = model_df[["entropy_std", "volatility_std"]]
y = model_df["crisis_next_4q"]

logit = LogisticRegression()
logit.fit(X, y)

coef_entropy = logit.coef_[0][0]
coef_vol = logit.coef_[0][1]

# ---------------------------------------
# SAVE DATASET
# ---------------------------------------
model_df.to_csv(os.path.join(folder_path, "fr_entropy_full_dataset.csv"), index=False)

print("FR_RESULTS folder fully saved.")
print("Entropy coefficient:", coef_entropy)
print("Volatility coefficient:", coef_vol)
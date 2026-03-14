import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# ---------------------------------------
# LOAD US DATA
# ---------------------------------------
us = pd.read_csv("US_RESULTS/us_entropy_full_dataset.csv")
us["country"] = "US"

# ---------------------------------------
# LOAD UK DATA
# ---------------------------------------
uk = pd.read_csv("UK_RESULTS/uk_entropy_full_dataset.csv")
uk["country"] = "UK"

# ---------------------------------------
# COMBINE INTO PANEL
# ---------------------------------------
panel = pd.concat([us, uk], ignore_index=True)

# Keep necessary columns
panel = panel[[
    "country",
    "date",
    "entropy_std",
    "volatility_std",
    "crisis_next_4q"
]]

# Drop any remaining missing
panel = panel.dropna().reset_index(drop=True)

print("Panel observations:", len(panel))

# ---------------------------------------
# POOLED LOGISTIC REGRESSION
# ---------------------------------------
X = panel[["entropy_std", "volatility_std"]]
y = panel["crisis_next_4q"]

logit = LogisticRegression()
logit.fit(X, y)

coef_entropy = logit.coef_[0][0]
coef_vol = logit.coef_[0][1]

print("PANEL RESULTS (US + UK)")
print("Entropy coefficient:", coef_entropy)
print("Volatility coefficient:", coef_vol)
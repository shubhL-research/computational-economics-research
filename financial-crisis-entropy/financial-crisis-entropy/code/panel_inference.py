import pandas as pd
import statsmodels.api as sm

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
# COMBINE PANEL
# ---------------------------------------
panel = pd.concat([us, uk], ignore_index=True)

panel = panel[[
    "country",
    "entropy_std",
    "volatility_std",
    "crisis_next_4q"
]]

panel = panel.dropna().reset_index(drop=True)

# ---------------------------------------
# LOGIT WITH STATS MODELS
# ---------------------------------------
X = panel[["entropy_std", "volatility_std"]]
X = sm.add_constant(X)

y = panel["crisis_next_4q"]

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

print(result.summary())
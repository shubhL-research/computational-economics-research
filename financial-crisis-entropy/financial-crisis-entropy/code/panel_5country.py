import pandas as pd
import statsmodels.api as sm

# LOAD DATA
us = pd.read_csv("US_RESULTS/us_entropy_full_dataset.csv")
us["country"] = "US"

uk = pd.read_csv("UK_RESULTS/uk_entropy_full_dataset.csv")
uk["country"] = "UK"

de = pd.read_csv("DE_RESULTS/de_entropy_full_dataset.csv")
de["country"] = "DE"

fr = pd.read_csv("FR_RESULTS/fr_entropy_full_dataset.csv")
fr["country"] = "FR"

ca = pd.read_csv("CA_RESULTS/ca_entropy_full_dataset.csv")
ca["country"] = "CA"

# COMBINE
panel = pd.concat([us, uk, de, fr, ca], ignore_index=True)

panel = panel[[
    "country",
    "entropy_std",
    "volatility_std",
    "crisis_next_4q"
]]

panel = panel.dropna().reset_index(drop=True)

print("Total observations:", len(panel))

# LOGIT
X = panel[["entropy_std", "volatility_std"]]
X = sm.add_constant(X)

y = panel["crisis_next_4q"]

model = sm.Logit(y, X)
result = model.fit(disp=False)

print(result.summary())
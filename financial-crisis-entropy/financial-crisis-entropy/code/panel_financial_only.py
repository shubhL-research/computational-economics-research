import pandas as pd
import statsmodels.api as sm
import os

# ==========================================
# FINANCIAL CRISIS COUNTRIES ONLY
# ==========================================

country_folders = [
    "US_RESULTS",
    "UK_RESULTS",
    "DE_RESULTS",
    "FR_RESULTS",
    "IT_RESULTS",
    "ES_RESULTS",
    "JP_RESULTS",
    "NL_RESULTS",
    "CA_RESULTS"
]

panel_list = []

for folder in country_folders:
    files = [f for f in os.listdir(folder) if f.endswith(".csv")]
    if len(files) == 0:
        continue

    file_path = os.path.join(folder, files[0])
    df = pd.read_csv(file_path)

    country = folder.replace("_RESULTS", "")
    df["country"] = country

    # Remove COVID crisis effect
    df = df[df["year"] != 2020]

    panel_list.append(df)

panel = pd.concat(panel_list, ignore_index=True)

print("Total observations:", len(panel))

# ==========================================
# FIXED EFFECTS
# ==========================================

country_dummies = pd.get_dummies(panel["country"], drop_first=True)

X = pd.concat([
    panel[["entropy_std", "volatility_std"]],
    country_dummies
], axis=1)

X = sm.add_constant(X)
y = panel["crisis_next_4q"]

X = X.astype(float)
y = y.astype(float)

# ==========================================
# LOGIT WITH CLUSTERED SE
# ==========================================

model = sm.Logit(y, X)

result = model.fit(
    cov_type="cluster",
    cov_kwds={"groups": panel["country"]},
    disp=False
)

print(result.summary())
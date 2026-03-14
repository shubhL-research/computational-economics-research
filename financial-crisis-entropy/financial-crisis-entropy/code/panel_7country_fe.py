import pandas as pd
import statsmodels.api as sm

countries = {
    "US": "US_RESULTS/us_entropy_full_dataset.csv",
    "UK": "UK_RESULTS/uk_entropy_full_dataset.csv",
    "DE": "DE_RESULTS/de_entropy_full_dataset.csv",
    "FR": "FR_RESULTS/fr_entropy_full_dataset.csv",
    "CA": "CA_RESULTS/ca_entropy_full_dataset.csv",
    "JP": "JP_RESULTS/jp_entropy_full_dataset.csv",
    "IT": "IT_RESULTS/it_full_dataset.csv",
    "ES": "ES_RESULTS/es_entropy_full_dataset.csv",
    "NL": "NL_RESULTS/nl_entropy_full_dataset.csv",
    "SE": "SE_RESULTS/se_entropy_full_dataset.csv"
}

dfs = []

for country, path in countries.items():
    df = pd.read_csv(path)
    df["country"] = country
    dfs.append(df)

panel = pd.concat(dfs, ignore_index=True)

panel = panel.dropna(subset=[
    "entropy_std",
    "volatility_std",
    "crisis_next_4q"
])

print("Total observations:", len(panel))

country_dummies = pd.get_dummies(panel["country"], drop_first=True)

X = pd.concat([
    panel[["entropy_std", "volatility_std"]],
    country_dummies
], axis=1)

X = sm.add_constant(X)
X = X.astype(float)

y = panel["crisis_next_4q"].astype(float)

model = sm.Logit(y, X)

result = model.fit(
    disp=False,
    cov_type="cluster",
    cov_kwds={"groups": panel["country"]}
)

print(result.summary())
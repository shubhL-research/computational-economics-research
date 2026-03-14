import pandas as pd
import numpy as np
import statsmodels.api as sm

# ----------------------------
# LOAD ENTROPY PANEL
# ----------------------------

panel = pd.read_csv("YOUR_EXISTING_PANEL_FILE.csv")
panel['date'] = pd.to_datetime(panel['date'])

# ----------------------------
# CREDIT FILES
# ----------------------------

credit_files = {
    'US': 'us_credit.csv',
    'UK': 'uk_credit.csv',
    'DE': 'de_credit.csv',
    'FR': 'fr_credit.csv',
    'IT': 'it_credit.csv',
    'ES': 'es_credit.csv',
    'NL': 'nl_credit.csv',
    'SE': 'se_credit.csv',
    'CA': 'ca_credit.csv',
    'JP': 'jp_credit.csv'
}

credit_list = []

for country, file in credit_files.items():

    df = pd.read_csv(file)
    credit_col = df.columns[1]

    df = df.rename(columns={
        'observation_date': 'date',
        credit_col: 'credit'
    })

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    df['credit_yoy'] = df['credit'].pct_change(4)
    df['country'] = country

    credit_list.append(df[['date','country','credit_yoy']])

credit_panel = pd.concat(credit_list)

# ----------------------------
# MERGE
# ----------------------------

panel = panel.merge(
    credit_panel,
    on=['date','country'],
    how='left'
)

panel = panel.dropna(subset=['credit_yoy'])

# ----------------------------
# STANDARDIZE
# ----------------------------

panel['credit_std'] = (
    (panel['credit_yoy'] - panel['credit_yoy'].mean())
    / panel['credit_yoy'].std()
)

# ----------------------------
# FIXED EFFECTS
# ----------------------------

dummies = pd.get_dummies(panel['country'], drop_first=True)

X = pd.concat([
    panel[['entropy_std','volatility_std','credit_std']],
    dummies
], axis=1)

X = sm.add_constant(X)

y = panel['crisis_next_8q']

# ----------------------------
# LOGIT WITH CLUSTER
# ----------------------------

model = sm.Logit(y, X)

result = model.fit(
    cov_type='cluster',
    cov_kwds={'groups': panel['country']}
)

print(result.summary())
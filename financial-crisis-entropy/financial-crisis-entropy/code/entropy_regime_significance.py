import pandas as pd
from scipy import stats

# =====================================================
# 📅 13 March 2026
# Entropy Regime Significance Testing
# =====================================================

panel = pd.read_csv("final_entropy_panel.csv")
panel['date'] = pd.to_datetime(panel['date'])

panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
].copy()

panel['crisis_now'] = panel['crisis_next_4q']
panel['crisis_shift'] = panel.groupby('country')['crisis_now'].diff()
panel['post_crisis'] = 0

for c in panel['country'].unique():
    country_df = panel[panel['country'] == c]
    ends = country_df[country_df['crisis_shift'] == -1]['date']
    for d in ends:
        mask = (
            (panel['country'] == c) &
            (panel['date'] > d) &
            (panel['date'] <= d + pd.DateOffset(months=12))
        )
        panel.loc[mask, 'post_crisis'] = 1

normal = panel[
    (panel['crisis_now'] == 0) &
    (panel['post_crisis'] == 0)
]['entropy_std']

crisis = panel[
    panel['crisis_now'] == 1
]['entropy_std']

post = panel[
    panel['post_crisis'] == 1
]['entropy_std']

print("\n=== T-Tests ===")

t1 = stats.ttest_ind(normal, crisis, equal_var=False)
t2 = stats.ttest_ind(normal, post, equal_var=False)

print("Normal vs Crisis p-value :", t1.pvalue)
print("Normal vs Post p-value   :", t2.pvalue)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 📅 13 March 2026
# Entropy Regime Comparison Analysis
# =====================================================

panel = pd.read_csv("final_entropy_panel.csv")
panel['date'] = pd.to_datetime(panel['date'])

panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
].copy()

# Crisis state today
panel['crisis_now'] = panel['crisis_next_4q']

# Identify post-crisis (1 quarter after crisis ends)
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

# Compute averages
normal_entropy = panel[
    (panel['crisis_now'] == 0) &
    (panel['post_crisis'] == 0)
]['entropy_std'].mean()

crisis_entropy = panel[
    panel['crisis_now'] == 1
]['entropy_std'].mean()

post_entropy = panel[
    panel['post_crisis'] == 1
]['entropy_std'].mean()

print("\n==== Average Entropy by Regime ====")
print("Normal Periods :", round(normal_entropy,3))
print("Crisis Periods :", round(crisis_entropy,3))
print("Post-Crisis    :", round(post_entropy,3))

# Bar plot
labels = ['Normal','Crisis','Post-Crisis']
values = [normal_entropy, crisis_entropy, post_entropy]

plt.figure(figsize=(6,4))
plt.bar(labels, values)
plt.title("Entropy Across Financial Regimes")
plt.ylabel("Average Standardized Entropy")
plt.show()
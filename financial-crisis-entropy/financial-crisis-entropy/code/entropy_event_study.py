import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 📅 15 March 2026
# Entropy Event Study (Physica A Build)
# =====================================================

panel = pd.read_csv("final_entropy_panel.csv")
panel['date'] = pd.to_datetime(panel['date'])

panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
].copy()

# Identify crisis start (0→1 transition)
panel['crisis_shift'] = panel.groupby('country')['crisis_next_4q'].diff()

event_windows = []

for country in panel['country'].unique():
    country_df = panel[panel['country'] == country].reset_index(drop=True)
    
    starts = country_df[country_df['crisis_shift'] == 1].index
    
    for idx in starts:
        for k in range(-8,9):
            if 0 <= idx+k < len(country_df):
                event_windows.append({
                    'relative_q': k,
                    'entropy': country_df.loc[idx+k,'entropy_std']
                })

event_df = pd.DataFrame(event_windows)

avg_path = event_df.groupby('relative_q')['entropy'].mean()

print("Total crisis events:", len(event_df['relative_q'].unique()))

plt.figure(figsize=(8,5))
plt.plot(avg_path.index, avg_path.values)
plt.axvline(0, linestyle='--')
plt.title("Average Entropy Around Crisis Start")
plt.xlabel("Quarters Relative to Crisis")
plt.ylabel("Average Entropy")
plt.show()
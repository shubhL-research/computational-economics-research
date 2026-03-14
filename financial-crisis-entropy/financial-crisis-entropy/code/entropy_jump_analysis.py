import pandas as pd
import numpy as np

# =====================================================
# 📅 15 March 2026
# Entropy Shock Magnitude Analysis
# =====================================================

panel = pd.read_csv("final_entropy_panel.csv")
panel['date'] = pd.to_datetime(panel['date'])

panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
].copy()

panel = panel.sort_values(['country','date'])
panel['crisis_shift'] = panel.groupby('country')['crisis_next_4q'].diff()

jumps = []

for country in panel['country'].unique():
    df = panel[panel['country'] == country].reset_index(drop=True)
    starts = df[df['crisis_shift'] == 1].index
    
    for idx in starts:
        if idx > 0 and idx+1 < len(df):
            pre = df.loc[idx-1,'entropy_std']
            at  = df.loc[idx,'entropy_std']
            post = df.loc[idx+1,'entropy_std']
            
            jumps.append({
                'jump_at_start': at - pre,
                'jump_next_q': post - at
            })

jumps_df = pd.DataFrame(jumps)

print("Average entropy jump at crisis start:",
      round(jumps_df['jump_at_start'].mean(),3))

print("Average entropy jump next quarter:",
      round(jumps_df['jump_next_q'].mean(),3))
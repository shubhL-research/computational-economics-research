# ============================================
# Critical Slowing Down Test (DEBUG VERSION)
# Date: 4 March 2026
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

panel = pd.read_csv("final_entropy_panel.csv")
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["country", "date"])

print("Total observations:", len(panel))

# Rolling variance
panel["entropy_var_8q"] = (
    panel.groupby("country")["entropy_std"]
    .rolling(8)
    .var()
    .reset_index(level=0, drop=True)
)

# Crisis onsets
panel["crisis_shift"] = panel.groupby("country")["crisis_next_4q"].diff()
events = panel[panel["crisis_shift"] == 1]

print("Crisis events found:", len(events))

window = 8
variance_series = []

for _, row in events.iterrows():
    country = row["country"]
    crisis_date = row["date"]

    subset = panel[panel["country"] == country].set_index("date")

    try:
        event_window = subset.loc[
            crisis_date - pd.DateOffset(months=3*window):
            crisis_date + pd.DateOffset(months=3*window)
        ]["entropy_var_8q"]

        if len(event_window) >= 5:
            variance_series.append(event_window.values)

    except:
        continue

print("Variance windows collected:", len(variance_series))

if len(variance_series) == 0:
    print("No variance windows found. Exiting.")
    exit()

min_len = min(len(s) for s in variance_series)
aligned = np.array([s[:min_len] for s in variance_series])
avg_variance = aligned.mean(axis=0)
tau = np.arange(-window, -window + min_len)

plt.figure()
plt.plot(tau, avg_variance)
plt.axvline(0)
plt.title("Entropy Variance Around Crisis Onset")
plt.xlabel("Event Time (Quarters)")
plt.ylabel("Rolling Variance (8Q)")
plt.tight_layout()

filename = "Figure5_CriticalSlowing.png"
plt.savefig(filename, dpi=300)
plt.show()

print("Figure saved at:", os.path.abspath(filename))
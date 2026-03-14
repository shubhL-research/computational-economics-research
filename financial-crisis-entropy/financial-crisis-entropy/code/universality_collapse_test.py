# ============================================
# Universality Collapse Test
# Date: 4 March 2026
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

panel = pd.read_csv("final_entropy_panel.csv")
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["country", "date"])

# Identify crisis onsets
panel["crisis_shift"] = panel.groupby("country")["crisis_next_4q"].diff()
events = panel[panel["crisis_shift"] == 1]

window = 8
normalized_paths = []

for _, row in events.iterrows():
    country = row["country"]
    crisis_date = row["date"]

    subset = panel[panel["country"] == country].set_index("date")

    try:
        event_window = subset.loc[
            crisis_date - pd.DateOffset(months=3*window):
            crisis_date + pd.DateOffset(months=3*window)
        ]["entropy_std"]

        if len(event_window) >= 10:

            values = event_window.values[:2*window+1]

            # Normalize relative to pre-crisis average (-4 to -1)
            pre_crisis_mean = np.mean(values[window-4:window])
            normalized = values - pre_crisis_mean

            normalized_paths.append(normalized)

    except:
        continue

if len(normalized_paths) == 0:
    print("No valid crisis windows found.")
    exit()

normalized_paths = np.array(normalized_paths)
tau = np.arange(-window, window+1)

# Plot individual normalized paths
plt.figure()

for path in normalized_paths:
    plt.plot(tau, path, alpha=0.3)

# Plot average collapse curve
mean_path = normalized_paths.mean(axis=0)
plt.plot(tau, mean_path, linewidth=3)

plt.axvline(0)
plt.title("Universality Collapse of Entropy Around Crisis")
plt.xlabel("Event Time (Quarters)")
plt.ylabel("Normalized Entropy")
plt.tight_layout()
plt.savefig("Figure6_UniversalityCollapse.png", dpi=300)
plt.show()

print("Universality collapse figure saved.")
print("Number of crisis events used:", len(normalized_paths))
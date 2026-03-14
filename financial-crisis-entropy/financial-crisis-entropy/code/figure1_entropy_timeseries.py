# ============================================
# FIGURE 1 — ENTROPY TIME SERIES (US)
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- PATHS ----------
DATA_PATH = r"C:\Users\rog\Desktop\OXFORD-HARVARD ECON PROJECT\Entropy Research\final_entropy_panel.csv"
SAVE_PATH = os.path.join(os.path.expanduser("~"), "Desktop")

# ---------- LOAD DATA ----------
panel = pd.read_csv(DATA_PATH)

# Force correct datetime parsing
panel["date"] = pd.to_datetime(panel["date"], errors="coerce")

# Drop rows where date failed to parse
panel = panel.dropna(subset=["date"])

# Sort properly
panel = panel.sort_values(["country", "date"])

print("Data loaded.")
print(panel[["date","country","entropy_std"]].head())

# ---------- FILTER UNITED STATES ----------
us = panel[panel["country"] == "United States"].copy()

if len(us) == 0:
    print("ERROR: No United States data found.")
    exit()

# ---------- PLOT ----------
plt.figure(figsize=(8,4))
plt.plot(us["date"], us["entropy_std"], color="black", linewidth=1.5)

plt.xlabel("Time")
plt.ylabel("Standardized Entropy")
plt.title("Entropy Time Series — United States")

plt.tight_layout()

plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_1_US_Entropy.png"), dpi=300)
plt.close()

print("Figure 1 saved to Desktop successfully.")
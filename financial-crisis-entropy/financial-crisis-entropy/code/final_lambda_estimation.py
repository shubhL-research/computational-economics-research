import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ============================================
# 1. LOAD DATA
# ============================================

DATA_PATH = r"C:\Users\rog\Desktop\OXFORD-HARVARD ECON PROJECT\Entropy Research\final_entropy_panel.csv"

panel = pd.read_csv(DATA_PATH, parse_dates=["date"])

print("Data loaded.")
print("Total observations:", len(panel))

# ============================================
# 2. SORT PROPERLY
# ============================================

panel = panel.sort_values(["country", "date"]).reset_index(drop=True)

# ============================================
# 3. IDENTIFY CRISIS EVENTS
# ============================================

# Assumes you already have a crisis indicator column
# If your column name is different, tell me.
if "crisis" not in panel.columns:
    raise ValueError("No 'crisis' column found in dataset.")

crisis_events = panel[panel["crisis"] == 1]

print("Crisis events found:", len(crisis_events))

# ============================================
# 4. BUILD EVENT WINDOWS
# ============================================

post_crisis_values = []

for idx in crisis_events.index:
    country = panel.loc[idx, "country"]
    crisis_date = panel.loc[idx, "date"]

    country_data = panel[panel["country"] == country].reset_index(drop=True)

    crisis_position = country_data[country_data["date"] == crisis_date].index

    if len(crisis_position) == 0:
        continue

    crisis_position = crisis_position[0]

    # collect 6 quarters after crisis
    window = country_data.iloc[crisis_position:crisis_position+6]

    if len(window) < 6:
        continue

    post_crisis_values.append(window["entropy_std"].values)

post_crisis_values = np.array(post_crisis_values)

print("Valid post-crisis windows:", len(post_crisis_values))

# ============================================
# 5. COMPUTE AVERAGE TRAJECTORY
# ============================================

mean_entropy = np.mean(post_crisis_values, axis=0)
tau = np.arange(len(mean_entropy))

# ============================================
# 6. DEFINE EXPONENTIAL MODEL
# ============================================

def exp_decay(t, A, lambd, C):
    return A * np.exp(-lambd * t) + C

# ============================================
# 7. FIT MODEL
# ============================================

params, _ = curve_fit(exp_decay, tau, mean_entropy)

A_est, lambda_est, C_est = params

print("\n===== FINAL RELAXATION ESTIMATE =====")
print("Lambda =", lambda_est)
print("A =", A_est)
print("C =", C_est)
print("=====================================")

# ============================================
# 8. OPTIONAL: VISUAL CHECK
# ============================================

plt.figure(figsize=(6,4))
plt.plot(tau, mean_entropy, 'o-', label="Average entropy")
plt.plot(tau, exp_decay(tau, *params), '--', label="Exponential fit")
plt.xlabel("Quarters After Crisis")
plt.ylabel("Entropy")
plt.legend()
plt.tight_layout()
plt.show()
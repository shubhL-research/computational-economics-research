# ============================================================
# FINAL JOURNAL-GRADE FIGURE GENERATION SCRIPT
# Saves: Final_Figure_1.png ... Final_Figure_6.png
# ============================================================

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

# ============================================================
# JOURNAL STYLE SETTINGS (Physica A style)
# ============================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "figure.dpi": 300,
    "savefig.dpi": 600
})

# ============================================================
# ASSUMES: panel DataFrame already exists
# Columns required:
# "date", "country", "entropy_std", "crisis_next_4q"
# ============================================================


# ============================================================
# FIGURE 1 — US ENTROPY TIME SERIES
# ============================================================

us = panel[panel["country"] == "US"].copy()

plt.figure(figsize=(6,3.8))
plt.plot(us["date"], us["entropy_std"], color="black")
plt.xlabel("Time")
plt.ylabel("Standardized Entropy")
plt.tight_layout()
plt.savefig("Final_Figure_1.png", dpi=600, bbox_inches="tight")
plt.close()


# ============================================================
# PREPARE EVENT-TIME DATA
# ============================================================

window = 12
event_curves = []

for c in panel["country"].unique():
    df_c = panel[panel["country"] == c].reset_index(drop=True)
    crisis_idx = df_c.index[df_c["crisis_next_4q"] == 1].tolist()

    for idx in crisis_idx:
        if idx - window >= 0 and idx + window < len(df_c):
            segment = df_c.loc[idx-window:idx+window, "entropy_std"].values
            event_curves.append(segment)

event_curves = np.array(event_curves)
tau = np.arange(-window, window+1)
entropy_avg = np.mean(event_curves, axis=0)


# ============================================================
# FIGURE 2 — EVENT STUDY AVERAGE
# ============================================================

plt.figure(figsize=(6,3.8))
plt.plot(tau, entropy_avg, color="black", linewidth=2)
plt.axvline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Event Time (τ)")
plt.ylabel("Standardized Entropy")
plt.tight_layout()
plt.savefig("Final_Figure_2.png", dpi=600, bbox_inches="tight")
plt.close()


# ============================================================
# FIGURE 3 — POST-CRISIS RELAXATION
# ============================================================

tau_post = tau[tau >= 0]
entropy_post = entropy_avg[tau >= 0]

def exp_decay(x, A, lam, C):
    return A * np.exp(-lam * x) + C

params, _ = curve_fit(exp_decay, tau_post, entropy_post, maxfev=10000)
A_est, lambda_est, C_est = params
fitted_curve = exp_decay(tau_post, A_est, lambda_est, C_est)

plt.figure(figsize=(6,3.8))
plt.plot(tau_post, entropy_post, color="black")
plt.plot(tau_post, fitted_curve, color="gray", linestyle="--")
plt.xlabel("Quarters Since Crisis (τ)")
plt.ylabel("Standardized Entropy")
plt.tight_layout()
plt.savefig("Final_Figure_3.png", dpi=600, bbox_inches="tight")
plt.close()

print("Estimated lambda:", lambda_est)


# ============================================================
# FIGURE 4 — PERSISTENCE BAR CHART
# ============================================================

def estimate_ar1(series):
    data = series.dropna().values
    y = data[1:]
    x = sm.add_constant(data[:-1])
    model = sm.OLS(y, x).fit()
    return model.params[1]

beta_full = estimate_ar1(panel["entropy_std"])
beta_normal = estimate_ar1(panel[panel["crisis_next_4q"] == 0]["entropy_std"])
beta_crisis = estimate_ar1(panel[panel["crisis_next_4q"] == 1]["entropy_std"])

plt.figure(figsize=(5,3.8))
plt.bar(["Full Sample", "Normal", "Crisis"],
        [beta_full, beta_normal, beta_crisis],
        color="black")
plt.ylabel("AR(1) Coefficient")
plt.tight_layout()
plt.savefig("Final_Figure_4.png", dpi=600, bbox_inches="tight")
plt.close()


# ============================================================
# FIGURE 5 — VARIANCE AMPLIFICATION
# ============================================================

panel["entropy_var8"] = panel.groupby("country")["entropy_std"] \
    .rolling(8).var().reset_index(level=0, drop=True)

event_var_curves = []

for c in panel["country"].unique():
    df_c = panel[panel["country"] == c].reset_index(drop=True)
    crisis_idx = df_c.index[df_c["crisis_next_4q"] == 1].tolist()

    for idx in crisis_idx:
        if idx - window >= 0 and idx + window < len(df_c):
            segment = df_c.loc[idx-window:idx+window, "entropy_var8"].values
            event_var_curves.append(segment)

event_var_curves = np.array(event_var_curves)
variance_avg = np.nanmean(event_var_curves, axis=0)

plt.figure(figsize=(6,3.8))
plt.plot(tau, variance_avg, color="black")
plt.axvline(0, color="gray", linestyle="--", linewidth=1)
plt.xlabel("Event Time (τ)")
plt.ylabel("Rolling Variance (8Q)")
plt.tight_layout()
plt.savefig("Final_Figure_5.png", dpi=600, bbox_inches="tight")
plt.close()


# ============================================================
# FIGURE 6 — CROSS-COUNTRY TRAJECTORIES
# ============================================================

plt.figure(figsize=(6,3.8))

for curve in event_curves:
    plt.plot(tau, curve, color="gray", alpha=0.35)

plt.plot(tau, entropy_avg, color="black", linewidth=2)
plt.axvline(0, color="gray", linestyle="--", linewidth=1)

plt.xlabel("Event Time (τ)")
plt.ylabel("Normalized Entropy")

plt.tight_layout()
plt.savefig("Final_Figure_6.png", dpi=600, bbox_inches="tight")
plt.close()

print("All FINAL figures generated successfully.")
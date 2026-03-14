# ============================================================
# PHYSICA A — FINAL FIGURE GENERATOR
# All Figures 1–6
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
import os

# ============================================================
# SETTINGS
# ============================================================

DATA_PATH = r"C:\Users\rog\Desktop\OXFORD-HARVARD ECON PROJECT\Entropy Research\final_entropy_panel.csv"
SAVE_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
WINDOW = 8
DPI = 300

plt.style.use("default")

print("Loading data...")
panel = pd.read_csv(DATA_PATH, parse_dates=["date"])
panel = panel.sort_values(["country", "date"])
print("Observations:", len(panel))


# ============================================================
# FIGURE 1 — US ENTROPY TIME SERIES
# ============================================================

us = panel[panel["country"] == "United States"]

plt.figure(figsize=(7,4))
plt.plot(us["date"], us["entropy_std"], color="black", linewidth=1)
plt.xlabel("Time")
plt.ylabel("Standardized Entropy")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_1.png"), dpi=DPI)
plt.close()

print("Figure 1 done.")


# ============================================================
# EVENT TIME CONSTRUCTION
# ============================================================

tau = np.arange(-WINDOW, WINDOW+1)
event_curves = []

for c in panel["country"].unique():

    df_c = panel[panel["country"] == c].reset_index(drop=True)
    crisis_idx = df_c.index[df_c["crisis_next_4q"] == 1].tolist()

    for idx in crisis_idx:
        if idx-WINDOW >= 0 and idx+WINDOW < len(df_c):

            segment = df_c.loc[idx-WINDOW:idx+WINDOW, "entropy_std"].values
            event_curves.append(segment)

event_curves = np.array(event_curves)
entropy_avg = np.nanmean(event_curves, axis=0)


# ============================================================
# FIGURE 2 — AVERAGE EVENT-TIME ENTROPY
# ============================================================

plt.figure(figsize=(6,4))

for curve in event_curves:
    plt.plot(tau, curve, color="gray", alpha=0.25)

plt.plot(tau, entropy_avg, color="black", linewidth=2)
plt.axvline(0, linestyle="--", color="black")
plt.xlabel("Event Time (τ)")
plt.ylabel("Standardized Entropy")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_2.png"), dpi=DPI)
plt.close()

print("Figure 2 done.")


# ============================================================
# FIGURE 3 — EXPONENTIAL RELAXATION (FROM PEAK)
# ============================================================

peak_index = np.argmax(entropy_avg)
tau_post = tau[peak_index:]
entropy_post = entropy_avg[peak_index:]

def exp_decay(t, A, lam, C):
    return A * np.exp(-lam * t) + C

# shift time to zero at peak
t_shifted = tau_post - tau_post[0]

params, _ = curve_fit(exp_decay, t_shifted, entropy_post, maxfev=10000)
A_hat, lambda_hat, C_hat = params

fit_curve = exp_decay(t_shifted, A_hat, lambda_hat, C_hat)

plt.figure(figsize=(6,4))
plt.plot(t_shifted, entropy_post, color="black", linewidth=2)
plt.plot(t_shifted, fit_curve, linestyle="--", color="black")
plt.xlabel("Quarters Since Peak Disorder")
plt.ylabel("Standardized Entropy")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_3.png"), dpi=DPI)
plt.close()

print("Figure 3 done.")
print("Estimated lambda:", round(lambda_hat,4))


# ============================================================
# FIGURE 4 — PANEL AR(1) PERSISTENCE
# ============================================================

def estimate_ar1_country(df):

    df = df.sort_values("date")
    series = df["entropy_std"].dropna().reset_index(drop=True)

    if len(series) < 6:
        return np.nan

    y = series.iloc[1:].values
    x = series.iloc[:-1].values
    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()

    return model.params[1]

betas_full = []
betas_normal = []
betas_crisis = []

for c in panel["country"].unique():

    df_c = panel[panel["country"] == c]

    betas_full.append(estimate_ar1_country(df_c))

    betas_normal.append(
        estimate_ar1_country(df_c[df_c["crisis_next_4q"] == 0])
    )

    betas_crisis.append(
        estimate_ar1_country(df_c[df_c["crisis_next_4q"] == 1])
    )

beta_full = np.nanmean(betas_full)
beta_normal = np.nanmean(betas_normal)
beta_crisis = np.nanmean(betas_crisis)

plt.figure(figsize=(5,4))
plt.bar(["Full","Normal","Crisis"],
        [beta_full, beta_normal, beta_crisis],
        color="black")
plt.ylabel("AR(1) Coefficient")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_4.png"), dpi=DPI)
plt.close()

print("Figure 4 done.")


# ============================================================
# FIGURE 5 — CRITICAL SLOWING (VARIANCE)
# ============================================================

panel["entropy_var8"] = panel.groupby("country")["entropy_std"] \
    .rolling(8).var().reset_index(level=0, drop=True)

event_var = []

for c in panel["country"].unique():

    df_c = panel[panel["country"] == c].reset_index(drop=True)
    crisis_idx = df_c.index[df_c["crisis_next_4q"] == 1].tolist()

    for idx in crisis_idx:
        if idx-WINDOW >= 0 and idx+WINDOW < len(df_c):

            segment = df_c.loc[idx-WINDOW:idx+WINDOW, "entropy_var8"].values
            event_var.append(segment)

event_var = np.array(event_var)
var_avg = np.nanmean(event_var, axis=0)

plt.figure(figsize=(6,4))
plt.plot(tau, var_avg, color="black", linewidth=2)
plt.axvline(0, linestyle="--", color="black")
plt.xlabel("Event Time (τ)")
plt.ylabel("Rolling Variance (8Q)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_5.png"), dpi=DPI)
plt.close()

print("Figure 5 done.")


# ============================================================
# FIGURE 6 — UNIVERSAL COLLAPSE (NORMALIZED)
# ============================================================

normalized_curves = []

for curve in event_curves:
    baseline = np.nanmean(curve[:WINDOW])
    normalized_curves.append(curve - baseline)

normalized_curves = np.array(normalized_curves)
normalized_avg = np.nanmean(normalized_curves, axis=0)

plt.figure(figsize=(6,4))

for curve in normalized_curves:
    plt.plot(tau, curve, color="gray", alpha=0.25)

plt.plot(tau, normalized_avg, color="black", linewidth=2)
plt.axvline(0, linestyle="--", color="black")
plt.xlabel("Event Time (τ)")
plt.ylabel("Normalized Entropy")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_6.png"), dpi=DPI)
plt.close()

print("Figure 6 done.")

print("ALL PHYSICA A FIGURES GENERATED SUCCESSFULLY.")
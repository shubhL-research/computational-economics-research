# ============================================================
# REMAINING FIGURES (4, 5, 6)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

SAVE_PATH = os.path.join(os.path.expanduser("~"), "Desktop")

# ============================================================
# FIGURE 4 — CORRECT PANEL AR(1)
# ============================================================

import numpy as np
import statsmodels.api as sm

def estimate_ar1_country(df):
    df = df.sort_values("date")
    series = df["entropy_std"].dropna().reset_index(drop=True)

    if len(series) < 5:
        return np.nan

    y = series.iloc[1:].values
    x = series.iloc[:-1].values

    x = sm.add_constant(x)

    model = sm.OLS(y, x).fit()

    return model.params[1]


# --- FULL SAMPLE ---
betas_full = []
for c in panel["country"].unique():
    df_c = panel[panel["country"] == c]
    betas_full.append(estimate_ar1_country(df_c))

beta_full = np.nanmean(betas_full)


# --- NORMAL PERIODS ---
betas_normal = []
for c in panel["country"].unique():
    df_c = panel[(panel["country"] == c) & (panel["crisis_next_4q"] == 0)]
    betas_normal.append(estimate_ar1_country(df_c))

beta_normal = np.nanmean(betas_normal)


# --- CRISIS PERIODS ---
betas_crisis = []
for c in panel["country"].unique():
    df_c = panel[(panel["country"] == c) & (panel["crisis_next_4q"] == 1)]
    betas_crisis.append(estimate_ar1_country(df_c))

beta_crisis = np.nanmean(betas_crisis)


# --- PLOT ---
import matplotlib.pyplot as plt
import os

SAVE_PATH = os.path.join(os.path.expanduser("~"), "Desktop")

plt.figure(figsize=(5,4))
plt.bar(["Full","Normal","Crisis"],
        [beta_full, beta_normal, beta_crisis],
        color="black")
plt.ylabel("AR(1) Coefficient")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_4.png"))
plt.close()

print("Figure 4 (fixed) saved successfully.")
# ============================================================
# FIGURE 4 — PERSISTENCE
# ============================================================

beta_full = estimate_ar1(panel["entropy_std"])
beta_normal = estimate_ar1(panel[panel["crisis_next_4q"]==0]["entropy_std"])
beta_crisis = estimate_ar1(panel[panel["crisis_next_4q"]==1]["entropy_std"])

plt.figure(figsize=(5,4))
plt.bar(["Full","Normal","Crisis"],
        [beta_full, beta_normal, beta_crisis],
        color="black")
plt.ylabel("AR(1) Coefficient")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_4.png"))
plt.close()

print("Figure 4 done.")

# ============================================================
# FIGURE 5 — ROLLING VARIANCE
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
plt.plot(tau, var_avg, color="black")
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("Event Time (τ)")
plt.ylabel("Rolling Variance (8Q)")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_5.png"))
plt.close()

print("Figure 5 done.")

# ============================================================
# FIGURE 6 — CROSS COUNTRY COLLAPSE
# ============================================================

plt.figure(figsize=(6,4))

for curve in event_curves:
    plt.plot(tau, curve, color="gray", alpha=0.3)

plt.plot(tau, entropy_avg, color="black", linewidth=2)
plt.axvline(0, color="gray", linestyle="--")
plt.xlabel("Event Time (τ)")
plt.ylabel("Normalized Entropy")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_PATH, "Final_Figure_6.png"))
plt.close()

print("Figure 6 done.")
print("ALL REMAINING FIGURES GENERATED SUCCESSFULLY.")
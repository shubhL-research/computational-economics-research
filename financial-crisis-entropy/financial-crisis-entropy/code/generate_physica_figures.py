import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit

# Load panel
panel = pd.read_csv("final_entropy_panel.csv")
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["country","date"])

# =====================================================
# FIGURE 1 — Entropy Time Series (US Example)
# =====================================================

us = panel[panel["country"] == "US"]

plt.figure()
plt.plot(us["date"], us["entropy_std"])
plt.title("Entropy Time Series — United States")
plt.xlabel("Date")
plt.ylabel("Standardized Entropy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Figure1_Entropy_TimeSeries.png", dpi=300)
plt.close()

# =====================================================
# FIGURE 2 — Event Study
# =====================================================

panel["crisis_shift"] = panel.groupby("country")["crisis_next_4q"].diff()
events = panel[panel["crisis_shift"] == 1]

window = 8
event_series = []

for _, row in events.iterrows():
    c = row["country"]
    t = row["date"]
    subset = panel[panel["country"] == c].set_index("date")
    try:
        event_window = subset.loc[
            t - pd.DateOffset(months=3*window):
            t + pd.DateOffset(months=3*window)
        ]["entropy_std"]
        if len(event_window) > 0:
            event_series.append(event_window.values)
    except:
        continue

min_len = min(len(s) for s in event_series)
aligned = np.array([s[:min_len] for s in event_series])
avg_entropy = aligned.mean(axis=0)
tau = np.arange(-window, -window + min_len)

plt.figure()
plt.plot(tau, avg_entropy)
plt.axvline(0)
plt.title("Average Entropy Around Crisis Onset")
plt.xlabel("Event Time (Quarters)")
plt.ylabel("Standardized Entropy")
plt.tight_layout()
plt.savefig("Figure2_EventStudy.png", dpi=300)
plt.close()

# =====================================================
# FIGURE 3 — Relaxation Fit
# =====================================================

def exp_decay(t, A, lam, C):
    return A * np.exp(-lam * t) + C

post_tau = tau[tau >= 0]
post_entropy = avg_entropy[tau >= 0]

params, _ = curve_fit(exp_decay, post_tau, post_entropy, maxfev=10000)
A_hat, lambda_hat, C_hat = params

t_fit = np.linspace(0, max(post_tau), 100)
fit_curve = exp_decay(t_fit, A_hat, lambda_hat, C_hat)

plt.figure()
plt.plot(post_tau, post_entropy)
plt.plot(t_fit, fit_curve)
plt.title("Post-Crisis Entropy Relaxation")
plt.xlabel("Quarters Since Crisis Onset")
plt.ylabel("Standardized Entropy")
plt.tight_layout()
plt.savefig("Figure3_Relaxation.png", dpi=300)
plt.close()

# =====================================================
# FIGURE 4 — Persistence Comparison
# =====================================================

panel["entropy_lag"] = panel.groupby("country")["entropy_std"].shift(1)

full = panel.dropna(subset=["entropy_std","entropy_lag"])
beta_full = sm.OLS(full["entropy_std"],
                   sm.add_constant(full["entropy_lag"])).fit().params[1]

normal = panel[panel["crisis_next_4q"] == 0].dropna(
    subset=["entropy_std","entropy_lag"])
beta_normal = sm.OLS(normal["entropy_std"],
                     sm.add_constant(normal["entropy_lag"])).fit().params[1]

crisis = panel[panel["crisis_next_4q"] == 1].dropna(
    subset=["entropy_std","entropy_lag"])
beta_crisis = sm.OLS(crisis["entropy_std"],
                     sm.add_constant(crisis["entropy_lag"])).fit().params[1]

plt.figure()
plt.bar(["Full Sample","Normal","Crisis"],
        [beta_full, beta_normal, beta_crisis])
plt.title("Entropy Persistence (AR(1) Coefficients)")
plt.ylabel("AR(1) Coefficient")
plt.tight_layout()
plt.savefig("Figure4_Persistence.png", dpi=300)
plt.close()

print("Figures saved successfully.")
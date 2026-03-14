import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# -----------------------
# PATH
# -----------------------
DATA_PATH = r"C:\Users\rog\Desktop\OXFORD-HARVARD ECON PROJECT\Entropy Research\final_entropy_panel.csv"

# -----------------------
# LOAD DATA
# -----------------------
panel = pd.read_csv(DATA_PATH)
panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
panel = panel.sort_values(["country", "date"])

WINDOW = 8
tau = np.arange(-WINDOW, WINDOW + 1)

event_curves = []

# -----------------------
# BUILD EVENT WINDOWS
# -----------------------
for c in panel["country"].unique():
    df_c = panel[panel["country"] == c].reset_index(drop=True)

    if "crisis_next_4q" not in df_c.columns:
        print("ERROR: crisis_next_4q column missing")
        exit()

    crisis_idx = df_c.index[df_c["crisis_next_4q"] == 1].tolist()

    for idx in crisis_idx:
        if idx - WINDOW >= 0 and idx + WINDOW < len(df_c):
            segment = df_c.loc[idx - WINDOW:idx + WINDOW, "entropy_std"].values
            event_curves.append(segment)

if len(event_curves) == 0:
    print("ERROR: No crisis events found.")
    exit()

event_curves = np.array(event_curves)

entropy_avg = np.nanmean(event_curves, axis=0)

# -----------------------
# FIND PEAK
# -----------------------
peak_index = np.argmax(entropy_avg)

tau_post = tau[peak_index:]
entropy_post = entropy_avg[peak_index:]

t_shifted = tau_post - tau_post[0]

# -----------------------
# EXPONENTIAL FIT
# -----------------------
def exp_decay(t, A, lam, C):
    return A * np.exp(-lam * t) + C

params, _ = curve_fit(exp_decay, t_shifted, entropy_post, maxfev=10000)

A_hat, lambda_hat, C_hat = params

print("\n===== RELAXATION ESTIMATE =====")
print("Lambda =", round(lambda_hat, 6))
print("A =", round(A_hat, 6))
print("C =", round(C_hat, 6))
print("================================\n")
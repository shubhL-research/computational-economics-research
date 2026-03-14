import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# =====================================================
# 📅 15 March 2026
# Entropy Relaxation Estimation
# =====================================================

panel = pd.read_csv("final_entropy_panel.csv")
panel['date'] = pd.to_datetime(panel['date'])

panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
].copy()

panel = panel.sort_values(['country','date'])
panel['crisis_shift'] = panel.groupby('country')['crisis_next_4q'].diff()

event_windows = []

for country in panel['country'].unique():
    df = panel[panel['country'] == country].reset_index(drop=True)
    starts = df[df['crisis_shift'] == 1].index
    
    for idx in starts:
        window = df.loc[idx:idx+8]  # 8 quarters after crisis
        if len(window) == 9:
            window = window[['entropy_std']].reset_index(drop=True)
            window['t'] = np.arange(0,9)
            event_windows.append(window)

# Average across events
all_events = pd.concat(event_windows)
avg_path = all_events.groupby('t')['entropy_std'].mean().reset_index()

# Define exponential decay model
def exp_decay(t, A, lambd, C):
    return C + A * np.exp(-lambd * t)

# Fit model
params, _ = curve_fit(exp_decay,
                      avg_path['t'],
                      avg_path['entropy_std'],
                      p0=[0.5, 0.5, 0])

A_est, lambda_est, C_est = params

print("\nEstimated Parameters:")
print("A (shock size):", round(A_est,4))
print("Lambda (decay rate):", round(lambda_est,4))
print("C (long-run level):", round(C_est,4))

# Plot
plt.plot(avg_path['t'], avg_path['entropy_std'], 'o', label="Data")
plt.plot(avg_path['t'],
         exp_decay(avg_path['t'], *params),
         label="Exponential Fit")
plt.xlabel("Quarters After Crisis")
plt.ylabel("Average Entropy")
plt.legend()
plt.show()
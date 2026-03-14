import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/learning_simulation_results.csv")

prices = df["price_firm1"]

# -----------------------------
# Compute entropy
# -----------------------------

counts = prices.value_counts()

probs = counts / counts.sum()

price_entropy = entropy(probs)

# -----------------------------
# Compute average price
# -----------------------------

avg_price = prices.mean()

cost = 10

# -----------------------------
# Regime classification
# -----------------------------

if avg_price <= cost + 2:

    regime = "Competitive"

elif price_entropy < 1.0:

    regime = "Collusive"

else:

    regime = "Chaotic"


print("Average Price:", avg_price)
print("Entropy:", price_entropy)
print("Detected Regime:", regime)


# -----------------------------
# Visualization
# -----------------------------

plt.figure(figsize=(6,4))

plt.bar(["Entropy"],[price_entropy])

plt.title("Market Entropy")

plt.savefig("figures/regime_entropy.png")

plt.show()

print("Regime classification complete.")
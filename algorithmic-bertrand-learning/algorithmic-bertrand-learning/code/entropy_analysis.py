import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# -------------------
# Load simulation data
# -------------------

df = pd.read_csv("../data/learning_simulation_results.csv")

prices = df["price_firm1"]

window = 100
entropy_values = []

# -------------------
# Rolling entropy
# -------------------

for i in range(len(prices) - window):

    segment = prices[i:i+window]

    counts = segment.value_counts()

    probabilities = counts / counts.sum()

    ent = entropy(probabilities)

    entropy_values.append(ent)

# -------------------
# Plot entropy
# -------------------

plt.figure(figsize=(10,5))

plt.plot(entropy_values)

plt.title("Entropy of Price Dynamics")

plt.xlabel("Time")
plt.ylabel("Entropy")

plt.savefig("../figures/price_entropy.png")

plt.show()

print("Entropy analysis complete.")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

os.makedirs("figures", exist_ok=True)

market_size = 100
cost = 10
T = 3000

price_grid = np.arange(10,50)

alpha = 0.1

epsilons = [0.01,0.05,0.1,0.2]

avg_prices = []
price_entropy = []

def demand(price_i, price_j):

    if price_i < price_j:
        return market_size
    elif price_i == price_j:
        return market_size/2
    else:
        return 0

def profit(price_i, price_j):

    q = demand(price_i, price_j)
    return (price_i - cost) * q


for epsilon in epsilons:

    Q1 = {p:0 for p in price_grid}
    Q2 = {p:0 for p in price_grid}

    prices = []

    for t in range(T):

        if np.random.rand() < epsilon:
            p1 = np.random.choice(price_grid)
        else:
            p1 = max(Q1,key=Q1.get)

        if np.random.rand() < epsilon:
            p2 = np.random.choice(price_grid)
        else:
            p2 = max(Q2,key=Q2.get)

        prof1 = profit(p1,p2)
        prof2 = profit(p2,p1)

        Q1[p1] = Q1[p1] + alpha*(prof1 - Q1[p1])
        Q2[p2] = Q2[p2] + alpha*(prof2 - Q2[p2])

        prices.append((p1+p2)/2)

    avg_prices.append(np.mean(prices))

    counts = pd.Series(prices).value_counts()
    probs = counts/counts.sum()
    price_entropy.append(entropy(probs))


# plot average price

plt.figure(figsize=(8,5))
plt.plot(epsilons,avg_prices,marker="o")
plt.xlabel("Exploration Rate (epsilon)")
plt.ylabel("Average Market Price")
plt.title("Exploration vs Market Price")
plt.savefig("figures/epsilon_price_experiment.png")
plt.show()

# plot entropy

plt.figure(figsize=(8,5))
plt.plot(epsilons,price_entropy,marker="o")
plt.xlabel("Exploration Rate (epsilon)")
plt.ylabel("Price Entropy")
plt.title("Exploration vs Price Entropy")
plt.savefig("figures/epsilon_entropy_experiment.png")
plt.show()

print("Experiment complete.")
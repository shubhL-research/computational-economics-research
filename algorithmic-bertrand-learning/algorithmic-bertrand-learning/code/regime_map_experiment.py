import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

os.makedirs("figures", exist_ok=True)

market_size = 100
cost = 10
T = 2000

price_grid = np.arange(10,50)

alpha = 0.1

epsilons = [0.005,0.01,0.02,0.05,0.1,0.2]


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


def simulate(epsilon):

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

        Q1[p1] += alpha*(prof1 - Q1[p1])
        Q2[p2] += alpha*(prof2 - Q2[p2])

        prices.append((p1+p2)/2)

    avg_price = np.mean(prices)

    counts = pd.Series(prices).value_counts()
    probs = counts/counts.sum()

    ent = entropy(probs)

    return avg_price, ent


regimes = []

for epsilon in epsilons:

    avg_price, ent = simulate(epsilon)

    if avg_price <= cost + 2:

        regime = "Competitive"

    elif ent < 1.5:

        regime = "Collusive"

    else:

        regime = "Chaotic"

    regimes.append(regime)

    print(epsilon, avg_price, ent, regime)


# visualize regime map

colors = []

for r in regimes:

    if r == "Competitive":
        colors.append("green")

    elif r == "Collusive":
        colors.append("blue")

    else:
        colors.append("red")


plt.figure(figsize=(8,5))

plt.scatter(epsilons,[1]*len(epsilons),c=colors,s=150)

plt.yticks([])

plt.xlabel("Exploration Rate (epsilon)")

plt.title("Market Regime Map")

plt.savefig("figures/regime_map.png")

plt.show()

print("Regime map created.")
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

market_size = 100
cost = 10
T = 2000

price_grid = np.arange(10,50)

epsilon = 0.05

learning_rates = [0.05,0.1,0.2]

runs = 100


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


def simulate(alpha):

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

    return np.mean(prices)


mean_prices = []

for alpha in learning_rates:

    results = []

    for r in range(runs):

        results.append(simulate(alpha))

    mean_prices.append(np.mean(results))


plt.figure(figsize=(8,5))

plt.plot(learning_rates,mean_prices,marker="o")

plt.xlabel("Learning Rate (alpha)")
plt.ylabel("Average Market Price")

plt.title("Effect of Learning Speed on Market Prices")

plt.savefig("figures/learning_rate_results.png")

plt.show()
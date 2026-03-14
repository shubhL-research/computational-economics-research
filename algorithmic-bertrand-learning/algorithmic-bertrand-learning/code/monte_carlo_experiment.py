import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

market_size = 100
cost = 10
T = 2000

price_grid = np.arange(10,50)

alpha = 0.1
runs = 50

epsilons = [0.01,0.05,0.1,0.2]


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

    return np.mean(prices)


mean_prices = []
std_prices = []

for epsilon in epsilons:

    experiment_results = []

    for r in range(runs):

        result = simulate(epsilon)
        experiment_results.append(result)

    mean_prices.append(np.mean(experiment_results))
    std_prices.append(np.std(experiment_results))


plt.figure(figsize=(8,5))

plt.errorbar(epsilons,mean_prices,yerr=std_prices,marker="o")

plt.xlabel("Exploration Rate (epsilon)")
plt.ylabel("Average Market Price")

plt.title("Monte Carlo Simulation Results")

plt.savefig("figures/monte_carlo_price.png")

plt.show()

print("Monte Carlo experiments complete.")
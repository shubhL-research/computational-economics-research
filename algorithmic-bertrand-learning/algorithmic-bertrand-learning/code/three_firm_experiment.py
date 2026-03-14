import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

market_size = 100
cost = 10
T = 2000

price_grid = np.arange(10,50)

alpha = 0.1
epsilon = 0.05

runs = 100


def demand(prices):

    min_price = min(prices)

    winners = [i for i,p in enumerate(prices) if p == min_price]

    demand_per_firm = market_size / len(winners)

    q = [0]*len(prices)

    for i in winners:
        q[i] = demand_per_firm

    return q


def simulate():

    Q = [{p:0 for p in price_grid} for _ in range(3)]

    prices_record = []

    for t in range(T):

        prices = []

        for i in range(3):

            if np.random.rand() < epsilon:
                p = np.random.choice(price_grid)
            else:
                p = max(Q[i], key=Q[i].get)

            prices.append(p)

        q = demand(prices)

        profits = [(prices[i]-cost)*q[i] for i in range(3)]

        for i in range(3):

            Q[i][prices[i]] += alpha*(profits[i] - Q[i][prices[i]])

        prices_record.append(np.mean(prices))

    return np.mean(prices_record)


results = []

for r in range(runs):

    results.append(simulate())


avg_price = np.mean(results)
std_price = np.std(results)

print("Average price (3 firms):", avg_price)
print("Std dev:", std_price)


plt.hist(results, bins=20)

plt.xlabel("Average Price")
plt.ylabel("Frequency")

plt.title("Price Distribution (3 Firms)")

plt.savefig("figures/three_firm_prices.png")

plt.show()
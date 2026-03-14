import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

market_size = 100
cost = 10
T = 2000

price_grid = np.arange(10,50)

alpha = 0.1
epsilon = 0.1


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


# -------------------------
# Reinforcement Learning
# -------------------------

def reinforcement_learning():

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


# -------------------------
# Mean Based Learning
# -------------------------

def mean_based_learning():

    payoff_sum1 = {p:0 for p in price_grid}
    payoff_sum2 = {p:0 for p in price_grid}

    count1 = {p:1 for p in price_grid}
    count2 = {p:1 for p in price_grid}

    prices = []

    for t in range(T):

        p1 = max(payoff_sum1,key=lambda p: payoff_sum1[p]/count1[p])
        p2 = max(payoff_sum2,key=lambda p: payoff_sum2[p]/count2[p])

        prof1 = profit(p1,p2)
        prof2 = profit(p2,p1)

        payoff_sum1[p1] += prof1
        payoff_sum2[p2] += prof2

        count1[p1] += 1
        count2[p2] += 1

        prices.append((p1+p2)/2)

    return np.mean(prices)


# -------------------------
# Run experiments
# -------------------------

rl_price = reinforcement_learning()
mean_price = mean_based_learning()

algorithms = ["Reinforcement Learning","Mean-Based Learning"]
prices = [rl_price,mean_price]


plt.figure(figsize=(8,5))

plt.bar(algorithms,prices)

plt.ylabel("Average Market Price")

plt.title("Algorithm Comparison")

plt.savefig("figures/algorithm_comparison.png")

plt.show()

print("Algorithm comparison complete.")
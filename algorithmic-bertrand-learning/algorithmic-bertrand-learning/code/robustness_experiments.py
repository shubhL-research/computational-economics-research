import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import os

os.makedirs("figures", exist_ok=True)

market_size = 100
T = 3000
price_grid = np.arange(10,50)

alpha = 0.1
epsilon = 0.1


# -------------------------
# Demand allocation
# -------------------------

def demand(prices):

    lowest_price = min(prices)
    winners = [i for i,p in enumerate(prices) if p == lowest_price]

    demand_share = market_size / len(winners)

    demand_list = [0]*len(prices)

    for i in winners:
        demand_list[i] = demand_share

    return demand_list


# -------------------------
# Profit function
# -------------------------

def profits(prices,costs):

    q = demand(prices)

    prof = []

    for i in range(len(prices)):

        prof.append((prices[i]-costs[i])*q[i])

    return prof


# -------------------------
# Simulation function
# -------------------------

def run_simulation(num_firms,costs):

    Q = [{p:0 for p in price_grid} for _ in range(num_firms)]

    prices_record = []

    for t in range(T):

        prices = []

        for i in range(num_firms):

            if np.random.rand() < epsilon:
                p = np.random.choice(price_grid)
            else:
                p = max(Q[i],key=Q[i].get)

            prices.append(p)

        prof = profits(prices,costs)

        for i in range(num_firms):

            Q[i][prices[i]] = Q[i][prices[i]] + alpha*(prof[i] - Q[i][prices[i]])

        prices_record.append(np.mean(prices))

    return prices_record


# -------------------------
# Experiment 1: two firms
# -------------------------

prices_two = run_simulation(2,[10,10])

# -------------------------
# Experiment 2: three firms
# -------------------------

prices_three = run_simulation(3,[10,10,10])

# -------------------------
# Experiment 3: asymmetric costs
# -------------------------

prices_asym = run_simulation(2,[10,15])


# -------------------------
# Plot comparison
# -------------------------

plt.figure(figsize=(10,5))

plt.plot(prices_two,label="2 Firms")
plt.plot(prices_three,label="3 Firms")
plt.plot(prices_asym,label="Asymmetric Costs")

plt.legend()

plt.title("Robustness Experiments: Market Structure Effects")

plt.savefig("figures/robustness_market_structure.png")

plt.show()

print("Robustness experiment complete.")
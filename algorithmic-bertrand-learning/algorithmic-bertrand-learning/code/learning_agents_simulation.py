import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create folders if missing
os.makedirs("../data", exist_ok=True)
os.makedirs("../figures", exist_ok=True)

# --------------------
# Market parameters
# --------------------

market_size = 100
cost = 10
T = 3000

price_grid = np.arange(10,50)

epsilon = 0.1  # exploration rate
alpha = 0.1    # learning rate

# --------------------
# Demand function
# --------------------

def demand(price_i, price_j):

    if price_i < price_j:
        return market_size
    
    elif price_i == price_j:
        return market_size/2
    
    else:
        return 0


# --------------------
# Profit function
# --------------------

def profit(price_i, price_j):

    q = demand(price_i, price_j)
    return (price_i - cost) * q


# --------------------
# Initialize learning tables
# --------------------

Q1 = {p:0 for p in price_grid}
Q2 = {p:0 for p in price_grid}

prices1 = []
prices2 = []
profits1 = []
profits2 = []

# --------------------
# Simulation
# --------------------

for t in range(T):

    # Firm 1 chooses price
    if np.random.rand() < epsilon:
        p1 = np.random.choice(price_grid)
    else:
        p1 = max(Q1, key=Q1.get)

    # Firm 2 chooses price
    if np.random.rand() < epsilon:
        p2 = np.random.choice(price_grid)
    else:
        p2 = max(Q2, key=Q2.get)

    # profits
    prof1 = profit(p1,p2)
    prof2 = profit(p2,p1)

    # update learning values
    Q1[p1] = Q1[p1] + alpha*(prof1 - Q1[p1])
    Q2[p2] = Q2[p2] + alpha*(prof2 - Q2[p2])

    prices1.append(p1)
    prices2.append(p2)
    profits1.append(prof1)
    profits2.append(prof2)


# --------------------
# Dataset
# --------------------

df = pd.DataFrame({
    "period":range(T),
    "price_firm1":prices1,
    "price_firm2":prices2,
    "profit_firm1":profits1,
    "profit_firm2":profits2
})

df.to_csv("../data/learning_simulation_results.csv",index=False)

print("Dataset saved.")


# --------------------
# Price plot
# --------------------

plt.figure(figsize=(10,5))

plt.plot(df["price_firm1"],label="Firm1")
plt.plot(df["price_firm2"],label="Firm2")

plt.legend()
plt.title("Learning Agent Pricing Dynamics")

plt.savefig("../figures/learning_price_paths.png")
plt.show()


# --------------------
# Profit plot
# --------------------

plt.figure(figsize=(10,5))

plt.plot(df["profit_firm1"],label="Firm1")
plt.plot(df["profit_firm2"],label="Firm2")

plt.legend()
plt.title("Profit Dynamics")

plt.savefig("../figures/learning_profit_paths.png")
plt.show()

print("Simulation complete.")
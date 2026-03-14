import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------
# Create folders automatically
# ---------------------------

os.makedirs("../data", exist_ok=True)
os.makedirs("../figures", exist_ok=True)

# ---------------------------
# Market parameters
# ---------------------------

market_size = 100
cost = 10
T = 500

price_min = 10
price_max = 50

# ---------------------------
# Demand function
# ---------------------------

def demand(price_i, price_j):

    if price_i < price_j:
        return market_size
    
    elif price_i == price_j:
        return market_size / 2
    
    else:
        return 0


# ---------------------------
# Profit function
# ---------------------------

def profit(price_i, price_j):

    q = demand(price_i, price_j)
    return (price_i - cost) * q


# ---------------------------
# Simulation
# ---------------------------

prices1 = []
prices2 = []
profits1 = []
profits2 = []

for t in range(T):

    p1 = np.random.randint(price_min, price_max)
    p2 = np.random.randint(price_min, price_max)

    prices1.append(p1)
    prices2.append(p2)

    profits1.append(profit(p1, p2))
    profits2.append(profit(p2, p1))


# ---------------------------
# Create dataset
# ---------------------------

df = pd.DataFrame({
    "period": range(T),
    "price_firm1": prices1,
    "price_firm2": prices2,
    "profit_firm1": profits1,
    "profit_firm2": profits2
})

df.to_csv("../data/random_pricing_results.csv", index=False)

print("Dataset saved.")

# ---------------------------
# Plot price paths
# ---------------------------

plt.figure(figsize=(10,5))

plt.plot(df["period"], df["price_firm1"], label="Firm 1")
plt.plot(df["period"], df["price_firm2"], label="Firm 2")

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Random Pricing in Bertrand Market")

plt.legend()

plt.savefig("../figures/random_price_paths.png")

plt.show()

print("Figure saved.")

# ---------------------------
# Plot profits
# ---------------------------

plt.figure(figsize=(10,5))

plt.plot(df["period"], df["profit_firm1"], label="Firm 1")
plt.plot(df["period"], df["profit_firm2"], label="Firm 2")

plt.xlabel("Time")
plt.ylabel("Profit")
plt.title("Profit Dynamics")

plt.legend()

plt.savefig("../figures/profit_paths.png")

plt.show()

print("Simulation complete.")
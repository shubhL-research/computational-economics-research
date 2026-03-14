import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------
# Load simulation data
# -----------------------------

df = pd.read_csv("data/learning_simulation_results.csv")

# -----------------------------
# Summary statistics
# -----------------------------

summary = {
    "Average Price Firm 1": df["price_firm1"].mean(),
    "Average Price Firm 2": df["price_firm2"].mean(),
    "Price Std Firm 1": df["price_firm1"].std(),
    "Price Std Firm 2": df["price_firm2"].std(),
    "Average Profit Firm 1": df["profit_firm1"].mean(),
    "Average Profit Firm 2": df["profit_firm2"].mean()
}

summary_df = pd.DataFrame(summary,index=[0])

summary_df.to_csv("data/summary_statistics.csv",index=False)

print("Summary statistics saved.")


# -----------------------------
# Price distribution plot
# -----------------------------

plt.figure(figsize=(8,5))

plt.hist(df["price_firm1"],bins=20,alpha=0.7,label="Firm1")
plt.hist(df["price_firm2"],bins=20,alpha=0.7,label="Firm2")

plt.legend()

plt.title("Distribution of Prices")

plt.xlabel("Price")
plt.ylabel("Frequency")

plt.savefig("figures/price_distribution.png")

plt.show()

print("Price distribution figure saved.")


# -----------------------------
# Profit distribution
# -----------------------------

plt.figure(figsize=(8,5))

plt.hist(df["profit_firm1"],bins=20,alpha=0.7,label="Firm1")
plt.hist(df["profit_firm2"],bins=20,alpha=0.7,label="Firm2")

plt.legend()

plt.title("Distribution of Profits")

plt.xlabel("Profit")
plt.ylabel("Frequency")

plt.savefig("figures/profit_distribution.png")

plt.show()

print("Profit distribution figure saved.")
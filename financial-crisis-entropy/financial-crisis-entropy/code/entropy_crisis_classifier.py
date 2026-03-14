import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

# =====================================================
# 📅 14 March 2026
# Entropy as Crisis Classifier
# =====================================================

# Load panel
panel = pd.read_csv("final_entropy_panel.csv")

# Convert date
panel['date'] = pd.to_datetime(panel['date'])

# Restrict to modern era (optional but cleaner)
panel = panel[
    (panel['date'] >= '1990-01-01') &
    (panel['date'] <= '2022-12-31')
].copy()

print("Observations:", len(panel))

# Drop missing just in case
panel = panel.dropna(subset=['entropy_std','crisis_next_4q'])

# =====================================================
# Logistic: Crisis ~ Entropy
# =====================================================

X = panel[['entropy_std']]
X = sm.add_constant(X)

y = panel['crisis_next_4q']

model = sm.Logit(y, X).fit()
print(model.summary())

# =====================================================
# AUC
# =====================================================

pred = model.predict(X)
auc = roc_auc_score(y, pred)

print("\n==============================")
print("Entropy-only AUC:", round(auc,4))
print("==============================")
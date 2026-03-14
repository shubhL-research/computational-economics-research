import pandas as pd

panel = pd.read_csv("final_entropy_panel.csv")
print(panel['date'].min())
print(panel['date'].max())
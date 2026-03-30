import matplotlib.pyplot as plt 
import pandas as pd

equity = pd.read_csv("equity_curve.csv", index_col=0, parse_dates=True)

plt.figure(figsize=(12,6))
plt.plot(equity)
plt.title("Equity Curve")
plt.show()
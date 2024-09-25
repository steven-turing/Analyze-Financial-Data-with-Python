# 2.Calculating Net Present Value
import codecademylib3_seaborn
import matplotlib.pyplot as plt

# Update Project A Cash Flows Here
project_a = [-1000000, 0, 0, 50000, 50000, 200000, 250000, 250000, 250000, 250000, 375000, 375000, 375000, 375000,
             375000, 250000, 250000, 250000, 250000, 100000]

# Update Project B Cash Flows Here
project_b = [-1000000, 50000, 50000, 50000, 50000, 250000, 500000, 500000, 500000, 500000, 100000, 100000, 100000,
             100000, 100000, 100000, 100000, 100000, 100000, 100000]

discount_rate = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125,
                 0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, 0.18]


def calculate_npv(rate, cash_flow):
    npv = 0
    for t in range(len(cash_flow)):
        npv += cash_flow[t] / (1 + rate) ** t
    return npv


npvs_a = list()
npvs_b = list()
for rate in discount_rate:
    npv_a = calculate_npv(rate, project_a)
    npvs_a.append(npv_a)
    npv_b = calculate_npv(rate, project_b)
    npvs_b.append(npv_b)

plt.plot(discount_rate, npvs_a, linewidth=2.0, color="red", label="Project A")
plt.plot(discount_rate, npvs_b, linewidth=2.0, color="blue", label="Project B")
plt.axhline(y=0, linewidth=0.5, color="black")
plt.title('NPV Profile for Projects A and B')
plt.xlabel('Discount Rate')
plt.ylabel('Net Present Value')
plt.legend()
plt.show()

# 3.Basic Stock Analysis
import codecademylib3_seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('AAPL_data.csv')
print(df.head())

df['Daily Log Rate of Return'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))

print(df['Daily Log Rate of Return'])

stdev = np.std(df['Daily Log Rate of Return'])
print(stdev)

plt.hist(df['Daily Log Rate of Return'].dropna())
plt.title('Histogram of AAPL Daily Log Rates of Return')
plt.xlabel('Log Rate of Return')
plt.ylabel('Number of Days')
plt.show()

# 3.Python Candlestick Chart
# A candlestick chart plots the daily opening price, closing price, lowest price and highest price of a particular stock, and shows how that price changed each day over a given period of
# time. Each day is shown by one “candlestick,” and can be helpful when analysts are trying to make predictions about how a particular price of a stock may move in the future.
import codecademylib3_seaborn
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc

df = pd.read_csv('AAPL_data.csv')
print(df.head())

df['Date'] = pd.to_datetime(df['Date'])
df["Date"] = df["Date"].apply(mdates.date2num)

candle_data = df[['Date', 'Open', 'High', 'Low', 'Close']]
print(candle_data.head())

f1, ax = plt.subplots(figsize = (10,5))
candlestick_ohlc(ax,candle_data.values, colorup='green', colordown='red')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.title('Candlestick Chart for AAPL')
plt.xlabel('Date')
plt.ylabel('Value($)')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt


bse_m = pd.read_csv('./BSE/M_STOCKS_IN_SENSEX.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bse_w = pd.read_csv('./BSE/W_STOCKS_IN_SENSEX.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bse_d = pd.read_csv('./BSE/D_STOCKS_IN_SENSEX.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
nse_m = pd.read_csv('./NSE/M_STOCKS_IN_NIFTY.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
nse_w = pd.read_csv('./NSE/W_STOCKS_IN_NIFTY.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
nse_d = pd.read_csv('./NSE/D_STOCKS_IN_NIFTY.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

BSE = [bse_m, bse_w, bse_d]
NSE = [nse_m, nse_w, nse_d]

Time_len = [len(bse_m.iloc[:, 0]), len(bse_w.iloc[:, 0]), len(bse_d.iloc[:, 0])]
intervals = ['Monthly', 'Weekly', 'Daily']
colors = ['orange', 'blue', 'green']
Market = [BSE, NSE]
Market_name = ['BSE', 'NSE']

for z in range(2):
    for i in range(10):
        for j in range(3):
            plt.plot(range(Time_len[j]), Market[z][j].iloc[:, i], color=colors[j])
            plt.xlabel(f'Time Points ({intervals[j]} basis)')
            plt.ylabel('Stock Prices')
            plt.title(f'Stock Prices vs Time ({intervals[j]}) for {Market_name[z]} - {Market[z][j].columns[i]}')
            plt.show()
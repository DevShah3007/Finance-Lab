import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tabulate import tabulate
import pandas as pd
from dateutil.relativedelta import relativedelta


def historcial_volatility(df,months):
    startDate = datetime(2022,12,1)
    endDate = datetime(2023,1,1)
    vol = []
    for j in range(months):
        idx = (df['Date'] >= startDate)
        startDate = startDate - relativedelta(months=1)
        last_month = df.loc[idx==True]
        # idx = last_month['Date'] <= endDate
        # last_month = last_month.loc[idx==True]
        for i in range(21):
            stock_price = last_month.iloc[:,i+1]
            arr = []
            for k in range(1,len(stock_price)):
                arr.append(np.log(stock_price.iloc[k]/stock_price.iloc[k-1]))
            vol.append(np.std(arr)*np.sqrt(252))
    return vol

    # print(last_month)

df = pd.read_csv("./bsedata1.csv")
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True)

vol = historcial_volatility(df,60)
# stock_name = [df.columns[i] for i in range(1,22)]
# ans = [[vol[i],stock_name[i]] for i in range(21)]
# print(tabulate(ans,headers='vol,sn'))
y_axis = [vol[i] for i in range(0,len(vol),21)]
x_axis = np.arange(1,61)
plt.plot(x_axis,y_axis)
plt.show()

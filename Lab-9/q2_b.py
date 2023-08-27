import numpy as np
from scipy.stats import norm
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None
plt.style.use('seaborn')
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def N(x):
    return norm.cdf(x)


def BSM(S0, sigma, K, T, r):
    if T == 0:
        call = max(0, S0 - K)
    else:
        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call = N(d1) * S0 - N(d2) * K * np.exp(-r * T)
    return call


def Vega(S0, sigma, K, T, r):
    d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return np.sqrt(T) * S0 * (1 / np.sqrt(2 * np.pi)) * np.exp(-(d1**2) / 2)


def implied_vol(S0, K, T, r, market_price, tol=0.00001):
    max_iter = 200  # max no. of iterations
    vol_old = 0.3  # initial guess
    for k in range(max_iter):
        bs_price = BSM(S0, vol_old, K, T, r)
        Cprime = Vega(S0, vol_old, K, T, r)
        C = bs_price - market_price
        vol_new = vol_old - C / Cprime
        if (abs(vol_old - vol_new) < tol):
            break
        vol_old = vol_new
    implied_vol = vol_new
    return implied_vol


def adjustTTM(x):
    y = x.days
    y = float(y)
    return y/365.0


r = 0.05


def dataframe_cleaning(df):
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    df['TTM'] = df['Maturity'] - df['Date']
    df.drop(columns=['Maturity', 'Date'], inplace=True)
    df['TTM'] = df['TTM'].apply(adjustTTM)


def implied_vol_plot(df, assetName):
    impvol = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        impvol[i] = implied_vol(df.iloc[i]['Underlying Value'], df.iloc[i]['Strike Price'], df.iloc[i]['TTM'], r, df.iloc[i]['Call Price'])
    df['Imp Vol'] = impvol
    df.dropna(inplace=True)

    filt = ((df['Imp Vol'] > 1) | (df['Imp Vol'] < 0))
    df = df.drop(index=df[filt].index)

    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Time to Maturity')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title(f'{assetName}: Implied Volatility vs (Strike Price, Time to Maturity)')
    ax1.scatter(df['Strike Price'], df['TTM'], df['Imp Vol'])
    ax1.view_init(elev=30, azim=50)
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.scatter(df['Strike Price'], df['Imp Vol'])
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Imp Vol')
    ax2.set_title(f'{assetName}: Implied Volatility vs Strike Price')
    plt.show()

    fig3, ax3 = plt.subplots()
    ax3.scatter(df['Imp Vol'], df['TTM'])
    ax3.set_xlabel('Time to Maturity')
    ax3.set_ylabel('Implied Volatility')
    ax3.set_title(f'{assetName}: Implied Volatility vs Time to Maturity')
    plt.show()


na_vals = '-'
df_nifty = pd.read_csv('NIFTYoptiondata.csv', na_values=na_vals)
df_nifty = df_nifty[df_nifty.index % 200 == 0]
dataframe_cleaning(df_nifty)
implied_vol_plot(df_nifty, 'NIFTY')

df_stock = pd.read_csv('stockoptiondata.csv', na_values=na_vals)
df_stock = df_stock[df_stock.index % 200 == 0]
dataframe_cleaning(df_stock)

df_hdfc = df_stock.loc[df_stock['Symbol'] == 'HDFCBANK']
implied_vol_plot(df_hdfc, 'HDFCBANK')

df_infy = df_stock.loc[df_stock['Symbol'] == 'INFY']
implied_vol_plot(df_infy, 'INFY')

df_infy = df_stock.loc[df_stock['Symbol'] == 'ITC']
implied_vol_plot(df_infy, 'ITC')

df_ril = df_stock.loc[df_stock['Symbol'] == 'RELIANCE']
implied_vol_plot(df_ril, 'RELIANCE')

df_titan = df_stock.loc[df_stock['Symbol'] == 'TITAN']
implied_vol_plot(df_titan, 'TITAN')
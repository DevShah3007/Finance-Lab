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
    d2 = d1 - sigma * np.sqrt(T)
    return np.sqrt(T) * K * np.exp(-r * T) * (1 / np.sqrt(2 * np.pi)) * np.exp(-(d2**2) / 2)


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
    return y / 365


r = 0.05


def dataframe_cleaning(df):
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    df['TTM'] = df['Maturity'] - df['Date']
    df.drop(columns=['Maturity'], inplace=True)
    df['TTM'] = df['TTM'].apply(adjustTTM)


def meanImpliedVolatility(df):
    impvol = np.zeros(df.shape[0])
    for i in range(df.shape[0]):
        impvol[i] = implied_vol(df.iloc[i]['Underlying Value'], df.iloc[i]['Strike Price'], df.iloc[i]['TTM'], r, df.iloc[i]['Call Price'])
    df['Imp Vol'] = impvol
    df.dropna(inplace=True)
    return np.mean(df['Imp Vol'])


def historicalVolatilityCalculator(data):
    Returns = data.pct_change()[1:]
    his_volatility = np.std(Returns) * np.sqrt(252)
    return his_volatility


na_vals = '-'
df_nifty = pd.read_csv('NIFTYoptiondata.csv', na_values=na_vals)
df_nifty = df_nifty[df_nifty.index % 100 == 0]
dataframe_cleaning(df_nifty)
df_stock = pd.read_csv('stockoptiondata.csv', na_values=na_vals)
dataframe_cleaning(df_stock)

df_hdfc = df_stock.loc[df_stock['Symbol'] == 'HDFCBANK']
df_hdfc = df_hdfc[df_hdfc.index % 100 == 0]

df_infy = df_stock.loc[df_stock['Symbol'] == 'INFY']
df_infy = df_infy[df_infy.index % 100 == 0]

df_itc = df_stock.loc[df_stock['Symbol'] == 'ITC']
df_itc = df_itc[df_itc.index % 100 == 0]

df_ril = df_stock.loc[df_stock['Symbol'] == 'RELIANCE']
df_ril = df_ril[df_ril.index % 100 == 0]

df_titan = df_stock.loc[df_stock['Symbol'] == 'TITAN']
df_titan = df_titan[df_titan.index % 100 == 0]


imp1 = meanImpliedVolatility(df_nifty)
imp2 = meanImpliedVolatility(df_hdfc)
imp3 = meanImpliedVolatility(df_infy)
imp4 = meanImpliedVolatility(df_itc)
imp5 = meanImpliedVolatility(df_ril)
imp6 = meanImpliedVolatility(df_titan)


df = pd.read_csv('./Underlying Data/NIFTY.csv')
his1 = historicalVolatilityCalculator(df['Underlying Value'])

df = pd.read_csv('./Underlying Data/HDFCBANK.csv')
his2 = historicalVolatilityCalculator(df['Underlying Value'])

df = pd.read_csv('./Underlying Data/INFY.csv')
his3 = historicalVolatilityCalculator(df['Underlying Value'])

df = pd.read_csv('./Underlying Data/ITC.csv')
his4 = historicalVolatilityCalculator(df['Underlying Value'])

df = pd.read_csv('./Underlying Data/RELIANCE.csv')
his5 = historicalVolatilityCalculator(df['Underlying Value'])

df = pd.read_csv('./Underlying Data/TITAN.csv')
his6 = historicalVolatilityCalculator(df['Underlying Value'])



df = pd.DataFrame()
impvol = [imp1, imp2, imp3, imp4,imp5,imp6]
hisvol = [his1, his2, his3, his4,his5,his6]
df['Underlying Asset'] = ['NIFTY','HDFCBANK', 'INFY', 'ITC', 'RELIANCE', 'TITAN']
df['Implied Volatility'] = impvol
df['Historical Volatility'] = hisvol
print(df)
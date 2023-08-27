import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def adjustTTM(x):
    y = x.days
    y = float(y)
    return y / 365


def dataframe_cleaning(df):
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    df['TTM'] = df['Maturity'] - df['Date']
    # df.drop(columns=['Maturity', 'Date'], inplace=True)
    df['TTM'] = df['TTM'].apply(adjustTTM)


def plot_maker(df, assetName, optionType):
    fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
    ax1.scatter(df['TTM'], df['Strike Price'], df[optionType],s=1)
    ax1.set_xlabel('Time to Maturity')
    ax1.set_ylabel('Strike Price')
    ax1.set_zlabel(f'{optionType}')
    ax1.set_title(f'{assetName}: {optionType} vs (Strike Price, Time to Maturity)')
    ax1.view_init(elev=10, azim=20)

    fig2, ax2 = plt.subplots()
    ax2.scatter(df['Strike Price'], df[optionType],s=1)
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel(f'{optionType}')
    ax2.set_title(f'{assetName}: {optionType} vs Strike Price')

    fig3, ax3 = plt.subplots()
    ax3.scatter(df['TTM'], df[optionType],s=1)
    ax3.set_xlabel('Time to Maturity')
    ax3.set_ylabel(f'{optionType}')
    ax3.set_title(f'{assetName}: {optionType} vs Time to Maturity')

    plt.show()


na_vals = '-'
df = pd.read_csv('NIFTYoptiondata.csv', na_values=na_vals)
dataframe_cleaning(df)
plot_maker(df, 'NIFTY', 'Call Price')
plot_maker(df, 'NIFTY', 'Put Price')

df_stock = pd.read_csv('stockoptiondata.csv', na_values=na_vals)
dataframe_cleaning(df_stock)

df_hdfc = df_stock.loc[df_stock['Symbol'] == 'HDFCBANK']
plot_maker(df_hdfc, 'HDFCBANK', 'Call Price')
plot_maker(df_hdfc, 'HDFCBANK', 'Put Price')

df_infy = df_stock.loc[df_stock['Symbol'] == 'INFY']
plot_maker(df_infy, 'INFY', 'Call Price')
plot_maker(df_infy, 'INFY', 'Put Price')

df_itc = df_stock.loc[df_stock['Symbol'] == 'ITC']
plot_maker(df_itc, 'ITC', 'Call Price')
plot_maker(df_itc, 'ITC', 'Put Price')

df_ril = df_stock.loc[df_stock['Symbol'] == 'RELIANCE']
plot_maker(df_ril, 'RELIANCE', 'Call Price')
plot_maker(df_ril, 'RELIANCE', 'Put Price')

df_titan = df_stock.loc[df_stock['Symbol'] == 'TITAN']
plot_maker(df_titan, 'TITAN', 'Call Price')
plot_maker(df_titan, 'TITAN', 'Put Price')
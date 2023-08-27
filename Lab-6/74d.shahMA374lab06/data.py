import yfinance as yf

stock_list1 = ['HDFCBANK.BO','ICICIBANK.BO','INFY.BO','LT.BO','ONGC.BO','RELIANCE.BO','TATAMOTORS.BO','TATASTEEL.BO','TCS.BO','WIPRO.BO']
stock_list2 = ['HBLPOWER.BO','ISGEC.BO','JPASSOCIAT.BO','MIRZAINT.BO','NBCC.BO','PNB.BO','PNBGILTS.BO','SUZLON.BO','UNITINT.BO','YESBANK.BO']
stock_list3 = ['HDFCBANK.NS','ICICIBANK.NS','INFY.NS','ITC.NS','LT.NS','RELIANCE.NS','SBIN.NS','TCS.NS','TITAN.NS','WIPRO.NS']
stock_list4 = ['AARTIDRUGS.NS','APARINDS.NS','BEL.NS','ECLERX.NS','HDFCLIFE.NS','LUXIND.NS','NHPC.NS','NOCIL.NS','RADICO.NS','VMART.NS']

print('stock_list:', stock_list1)
data = yf.download(stock_list1, start="2018-01-01", end="2022-12-31", interval='1d')
data.to_csv('./BSE/D_STOCKS_IN_SENSEX.csv')

print('stock_list:', stock_list2)
data = yf.download(stock_list2, start="2018-01-01", end="2022-12-31", interval='1d')
data.to_csv('./BSE/D_STOCKS_NOT_IN_SENSEX.csv')

print('stock_list:', stock_list3)
data = yf.download(stock_list3, start="2018-01-01", end="2022-12-31", interval='1d')
data.to_csv('./NSE/D_STOCKS_IN_NIFTY.csv')

print('stock_list:', stock_list4)
data = yf.download(stock_list4, start="2018-01-01", end="2022-12-31", interval='1d')
data.to_csv('./NSE/D_STOCKS_NOT_IN_NIFTY.csv')

print('NSE')
data = yf.download('^NSEI', start="2018-01-01", end="2022-12-31", interval='1d')
data.to_csv('./NSE/D_NIFTY.csv')

print('BSE')
data = yf.download('^BSESN', start="2018-01-01", end="2022-02-01", interval='1d')
data.to_csv('./BSE/D_SENSEX.csv')
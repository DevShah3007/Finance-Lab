import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
sns.set(style="darkgrid")

image_count = 0

def compute_d1(sigma,T,t,St,K,r):
    if t >= T:
        return 0
    return (np.log(St/K) + (r+(sigma*sigma)/2.0)*(T-t))/(sigma*np.sqrt(T-t))

def compute_d2(sigma,T,t,St,K,r):
    if t >= T:
        return 0
    return (np.log(St/K) + (r-(sigma*sigma)/2.0)*(T-t))/(sigma*np.sqrt(T-t))

def compute_c(sigma,T,t,St,K,r):
    if t >= T:
        return max(0,St-K)
    d1 = compute_d1(sigma,T,t,St,K,r)
    d2 = compute_d2(sigma,T,t,St,K,r)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    return St*N_d1 - K*np.e**(-r*(T-t))*N_d2

def compute_p(sigma,T,t,St,K,r):
    if t >= T:
        return max(0,K-St)
    return compute_c(sigma,T,t,St,K,r) - St + K*np.e**(-r*(T-t))

def get_historical_volatility(stock_type):
    idx = stock_type['Date'] >= '2022-12-01'
    data_last_month = stock_type.loc[idx==True]
    volatility = np.zeros(21)
    for i in range(21):
        price = data_last_month.iloc[:,i+1]
        log_returns = []
        for j in range(1,len(price)):
            log_returns.append(np.log(price.iloc[j]/price.iloc[j-1]))
        # returns = data_last_month.iloc[:,i+1].pct_change()[1:]
        sigma = np.std(log_returns)*np.sqrt(252)
        volatility[i] = sigma
    return volatility

print()
print("PART-A")

bse = pd.read_csv('./bsedata1.csv')
nse = pd.read_csv('./nsedata1.csv')

bse['Date'] = pd.to_datetime(bse['Date'],dayfirst=True)
nse['Date'] = pd.to_datetime(nse['Date'],dayfirst=True)

print()
print("For Stocks in bsedata1:")
print()
stock_name_bse = [col for col in bse.columns][1:]
annualised_volatility_bse = get_historical_volatility(bse)
answer = []
for i in range(len(stock_name_bse)):
    answer.append([stock_name_bse[i],annualised_volatility_bse[i]])
print(tabulate(answer,headers=['Stock Name','Annualised Volatility']))


print()
print()
print("For Stocks in nsedata1:")
print()
stock_name_nse = [col for col in nse.columns][1:]
annualised_volatility_nse = get_historical_volatility(nse)
answer = []
for i in range(len(stock_name_nse)):
    answer.append([stock_name_nse[i],annualised_volatility_nse[i]])
print(tabulate(answer,headers=['Stock Name','Annualised Volatility']))


print()
print()
print()
print("PART-B")

final_stock_price_bse = [bse.iloc[-1,1:][i] for i in range(20)]
final_stock_price_nse = [nse.iloc[-1,1:][i] for i in range(20)]

call_option_price_bse = [compute_c(annualised_volatility_bse[i],0.5,0,final_stock_price_bse[i],final_stock_price_bse[i],0.05) for i in range(20)]
put_option_price_bse = [compute_p(annualised_volatility_bse[i],0.5,0,final_stock_price_bse[i],final_stock_price_bse[i],0.05) for i in range(20)]
call_option_price_nse = [compute_c(annualised_volatility_nse[i],0.5,0,final_stock_price_nse[i],final_stock_price_nse[i],0.05) for i in range(20)]
put_option_price_nse = [compute_p(annualised_volatility_nse[i],0.5,0,final_stock_price_nse[i],final_stock_price_nse[i],0.05) for i in range(20)]

answer_bse = []
answer_nse = []
for i in range(20):
    answer_bse.append([stock_name_bse[i],call_option_price_bse[i],put_option_price_bse[i]])
    answer_nse.append([stock_name_nse[i],call_option_price_nse[i],put_option_price_nse[i]])


print()
print("For Stocks in bsedata1:")
print()
print(tabulate(answer_bse,headers=['Stock Name','6 month Call-Option Price','6 month Put-Option Price']))
print()
print()
print("For Stocks in nsedata1:")
print()
print(tabulate(answer_nse,headers=['Stock Name','6 month Call-Option Price','6 month Put-Option Price']))


# print()
# print()

# for A in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
#     call_option_price_bse = [compute_c(annualised_volatility_bse[i],0.5,0,final_stock_price_bse[i],A*final_stock_price_bse[i],0.05) for i in range(20)]
#     put_option_price_bse = [compute_p(annualised_volatility_bse[i],0.5,0,final_stock_price_bse[i],A*final_stock_price_bse[i],0.05) for i in range(20)]
#     call_option_price_nse = [compute_c(annualised_volatility_nse[i],0.5,0,final_stock_price_nse[i],A*final_stock_price_nse[i],0.05) for i in range(20)]
#     put_option_price_nse = [compute_p(annualised_volatility_nse[i],0.5,0,final_stock_price_nse[i],A*final_stock_price_nse[i],0.05) for i in range(20)]

#     answer_bse = []
#     answer_nse = []
#     for i in range(20):
#         answer_bse.append([stock_name_bse[i],call_option_price_bse[i],put_option_price_bse[i]])
#         answer_nse.append([stock_name_nse[i],call_option_price_nse[i],put_option_price_nse[i]])

#     print()
#     print()
#     print("----- For K = ",A, "*S0 -----")
#     print()
#     print("For Stocks in bsedata1:")
#     print()
#     print(tabulate(answer_bse,headers=['Stock Name','6 month Call-Option Price','6 month Put-Option Price']))
#     print()
#     print()
#     print("For Stocks in nsedata1:")
#     print()
#     print(tabulate(answer_nse,headers=['Stock Name','6 month Call-Option Price','6 month Put-Option Price']))


# print()
# print()
# print()
# print("PART-C")

# row = 2
# col = 5

# # For volatility
# for j in range(2):
#     fig, axs = plt.subplots(row,col)
#     for num_stock in range(j*10,(j+1)*10,1):
#         x_axis = []
#         y_volatility = []
#         y_call_option = []
#         y_put_option = []
#         startDate = datetime(2023,1,1)
#         for i in range(60):
#             x_axis.append(i+1)
#             startDate = startDate - relativedelta(months=1)
#             idx = bse['Date'] >= startDate
#             data_last_month = bse.loc[idx==True]
#             price = data_last_month.iloc[:,num_stock+1]
#             log_returns = []
#             for j in range(1,len(price)):
#                 log_returns.append(np.log(price.iloc[j]/price.iloc[j-1]))
#             # returns = data_last_month.iloc[:,num_stock+1].pct_change()[1:]
#             sigma = np.std(log_returns)*np.sqrt(252)
#             y_volatility.append(sigma)
#         axs[(num_stock//col)%row][num_stock%col].plot(x_axis,y_volatility,color='green')
#         axs[(num_stock//col)%row][num_stock%col].set_xlabel('Months')
#         axs[(num_stock//col)%row][num_stock%col].set_ylabel('Volatility')
#         axs[(num_stock//col)%row][num_stock%col].set_title(f'{stock_name_bse[num_stock]}')
#     fig.suptitle(f'Volatility v/s Number of Months for Stocks in BSEdata1')
#     # plt.tight_layout()
#     # manager = plt.get_current_fig_manager()
#     # manager.resize(*manager.window.maxsize())
#     # plt.savefig(f'./part-c/{image_count}.png')
#     # image_count += 1
#     plt.subplots_adjust(left=0.07,bottom=0.08,right=0.99,top=0.90,wspace=0.7,hspace=0.4)
#     plt.show()

# for j in range(2):
#     fig, axs = plt.subplots(row,col)
#     for num_stock in range(j*10,(j+1)*10,1):
#         x_axis = []
#         y_volatility = []
#         y_call_option = []
#         y_put_option = []
#         startDate = datetime(2023,1,1)
#         for i in range(60):
#             x_axis.append(i+1)
#             startDate = startDate - relativedelta(months=1)
#             idx = nse['Date'] >= startDate
#             data_last_month = nse.loc[idx==True]
#             price = data_last_month.iloc[:,num_stock+1]
#             log_returns = []
#             for j in range(1,len(price)):
#                 log_returns.append(np.log(price.iloc[j]/price.iloc[j-1]))
#             # returns = data_last_month.iloc[:,num_stock+1].pct_change()[1:]
#             sigma = np.std(log_returns)*np.sqrt(252)
#             y_volatility.append(sigma)
#         axs[(num_stock//col)%row][num_stock%col].plot(x_axis,y_volatility,color='green')
#         axs[(num_stock//col)%row][num_stock%col].set_xlabel('Months')
#         axs[(num_stock//col)%row][num_stock%col].set_ylabel('Volatility')
#         axs[(num_stock//col)%row][num_stock%col].set_title(f'{stock_name_nse[num_stock]}')
#     fig.suptitle(f'Volatility v/s Number of Months for Stocks in NSEdata1')
#     # plt.tight_layout()
#     # manager = plt.get_current_fig_manager()
#     # manager.resize(*manager.window.maxsize())
#     # plt.savefig(f'./part-c/{image_count}.png')
#     # image_count += 1
#     plt.subplots_adjust(left=0.07,bottom=0.08,right=0.99,top=0.90,wspace=0.7,hspace=0.4)
#     plt.show()


# row = 2
# col = 5

# for A in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
#     for j in range(4):
#         fig, axs = plt.subplots(row,col)
#         for num_stock in range(j*5,(j+1)*5,1):
#             x_axis = []
#             y_volatility = []
#             y_call_option = []
#             y_put_option = []
#             startDate = datetime(2023,1,1)
#             for i in range(60):
#                 x_axis.append(i+1)
#                 startDate = startDate - relativedelta(months=1)
#                 idx = bse['Date'] >= startDate
#                 data_last_month = bse.loc[idx==True]
#                 price = data_last_month.iloc[:,num_stock+1]
#                 log_returns = []
#                 for j in range(1,len(price)):
#                     log_returns.append(np.log(price.iloc[j]/price.iloc[j-1]))
#                 # returns = data_last_month.iloc[:,num_stock+1].pct_change()[1:]
#                 sigma = np.std(log_returns)*np.sqrt(252)
#                 y_volatility.append(sigma)
#                 y_call_option.append(compute_c(sigma,0.5,0,final_stock_price_bse[num_stock],A*final_stock_price_bse[num_stock],0.05))
#                 y_put_option.append(compute_p(sigma,0.5,0,final_stock_price_bse[num_stock],A*final_stock_price_bse[num_stock],0.05))
#             axs[0][num_stock%col].plot(x_axis,y_call_option,color = 'red')
#             axs[0][num_stock%col].set_xlabel('Months')
#             axs[0][num_stock%col].set_ylabel('Call Price')
#             axs[0][num_stock%col].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#             axs[0][num_stock%col].set_title(f'{stock_name_bse[num_stock]}')
#             axs[1][num_stock%col].plot(x_axis,y_put_option,color = 'blue')
#             axs[1][num_stock%col].set_xlabel('Months')
#             axs[1][num_stock%col].set_ylabel('Put Price')
#             axs[1][num_stock%col].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#             axs[1][num_stock%col].set_title(f'{stock_name_bse[num_stock]}')
#         fig.suptitle(f'A = {A} \t Option Price v/s Number of Months for Stocks in BSEdata1')
#         # plt.tight_layout()
#         # manager = plt.get_current_fig_manager()
#         # manager.resize(*manager.window.maxsize())
#         # plt.savefig(f'./part-c/{image_count}.png')
#         # image_count += 1
#         plt.subplots_adjust(left=0.07,bottom=0.08,right=0.99,top=0.90,wspace=0.7,hspace=0.4)
#         plt.show()


# row = 2
# col = 5

# for A in [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]:
#     for j in range(4):
#         fig, axs = plt.subplots(row,col)
#         for num_stock in range(j*5,(j+1)*5,1):
#             x_axis = []
#             y_volatility = []
#             y_call_option = []
#             y_put_option = []
#             startDate = datetime(2023,1,1)
#             for i in range(60):
#                 x_axis.append(i+1)
#                 startDate = startDate - relativedelta(months=1)
#                 idx = nse['Date'] >= startDate
#                 data_last_month = nse.loc[idx==True]
#                 price = data_last_month.iloc[:,num_stock+1]
#                 log_returns = []
#                 for j in range(1,len(price)):
#                     log_returns.append(np.log(price.iloc[j]/price.iloc[j-1]))
#                 # returns = data_last_month.iloc[:,num_stock+1].pct_change()[1:]
#                 sigma = np.std(log_returns)*np.sqrt(252)
#                 y_volatility.append(sigma)
#                 y_call_option.append(compute_c(sigma,0.5,0,final_stock_price_nse[num_stock],A*final_stock_price_nse[num_stock],0.05))
#                 y_put_option.append(compute_p(sigma,0.5,0,final_stock_price_nse[num_stock],A*final_stock_price_nse[num_stock],0.05))
#             axs[0][num_stock%col].plot(x_axis,y_call_option,color = 'red')
#             axs[0][num_stock%col].set_xlabel('Months')
#             axs[0][num_stock%col].set_ylabel('Call Price')
#             axs[0][num_stock%col].set_title(f'{stock_name_nse[num_stock]}')
#             axs[0][num_stock%col].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#             axs[1][num_stock%col].plot(x_axis,y_put_option,color = 'blue')
#             axs[1][num_stock%col].set_xlabel('Months')
#             axs[1][num_stock%col].set_ylabel('Put Price')
#             axs[1][num_stock%col].set_title(f'{stock_name_nse[num_stock]}')
#             axs[1][num_stock%col].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#         fig.suptitle(f'A = {A} \t Option Price v/s Number of Months for Stocks in NSEdata1')
#         # plt.tight_layout()
#         # manager = plt.get_current_fig_manager()
#         # manager.resize(*manager.window.maxsize())
#         # plt.savefig(f'./part-c/{image_count}.png')
#         # image_count += 1
#         plt.subplots_adjust(left=0.07,bottom=0.08,right=0.99,top=0.90,wspace=0.7,hspace=0.4)
#         plt.show()

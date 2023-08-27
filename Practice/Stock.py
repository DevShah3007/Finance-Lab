import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pandas as pd

df = pd.read_csv('Stock.csv',usecols=[1,2,3,4,5,6,7,8,9,10])
df = df.pct_change().iloc[1:,]
M = df.mean()
C = df.cov()
u = np.array([1 for i in range(len(M))])

# w_min = (u@np.linalg.inv(C))/(u@np.linalg.inv(C)@np.transpose(u))
# sigma_min = w_min@C@np.transpose(w_min)
# u_min = M@np.transpose(w_min)
# print(u_min)
# print(sigma_min)

# markovitz_return = []
# markovitz_risk = []
# non_markovitz_return = []
# non_markovitz_risk = []

# y_axis = np.linspace(-2.0,2.0,5000)

# for uv in y_axis:
#     d1 = np.linalg.det([[1,u@np.linalg.inv(C)@np.transpose(M)],[uv,M@np.linalg.inv(C)@np.transpose(M)]])
#     d2 = np.linalg.det([[u@np.linalg.inv(C)@np.transpose(u),1],[M@np.linalg.inv(C)@np.transpose(u),uv]])
#     d3 = np.linalg.det([[u@np.linalg.inv(C)@np.transpose(u),u@np.linalg.inv(C)@np.transpose(M)],[M@np.linalg.inv(C)@np.transpose(u),M@np.linalg.inv(C)@np.transpose(M)]])
#     w = (d1*(u@np.linalg.inv(C)) + d2*(M@np.linalg.inv(C)))/d3
#     mu = M@np.transpose(w)
#     sigma = w@C@np.transpose(w)
    
#     if mu>=u_min:
#         markovitz_return.append(mu)
#         markovitz_risk.append(math.sqrt(sigma))
#     else:
#         non_markovitz_return.append(mu)
#         non_markovitz_risk.append(math.sqrt(sigma))

# plt.plot(markovitz_risk,markovitz_return,label='MEB',c='Black')
# plt.plot(non_markovitz_risk,non_markovitz_return,c='Red')
# plt.legend()
# plt.xlabel("Risk")
# plt.ylabel("Return")
# plt.show()

# # func1 = interpolate.interp1d(markovitz_risk,markovitz_return)
# # max_return = func1(0.15)
# # d1 = np.linalg.det([[1,u@np.linalg.inv(C)@np.transpose(M)],[max_return,M@np.linalg.inv(C)@np.transpose(M)]])
# # d2 = np.linalg.det([[u@np.linalg.inv(C)@np.transpose(u),1],[M@np.linalg.inv(C)@np.transpose(u),max_return]])
# # d3 = np.linalg.det([[u@np.linalg.inv(C)@np.transpose(u),u@np.linalg.inv(C)@np.transpose(M)],[M@np.linalg.inv(C)@np.transpose(u),M@np.linalg.inv(C)@np.transpose(M)]])
# # w = (d1*(u@np.linalg.inv(C)) + d2*(M@np.linalg.inv(C)))/d3
# # sigma = w@C@np.transpose(w)
# # print(max_return)
# # print(w)

# # func2 = interpolate.interp1d(non_markovitz_risk,non_markovitz_return)
# # min_return = func2(0.15)
# # d1 = np.linalg.det([[1,u@np.linalg.inv(C)@np.transpose(M)],[min_return,M@np.linalg.inv(C)@np.transpose(M)]])
# # d2 = np.linalg.det([[u@np.linalg.inv(C)@np.transpose(u),1],[M@np.linalg.inv(C)@np.transpose(u),min_return]])
# # d3 = np.linalg.det([[u@np.linalg.inv(C)@np.transpose(u),u@np.linalg.inv(C)@np.transpose(M)],[M@np.linalg.inv(C)@np.transpose(u),M@np.linalg.inv(C)@np.transpose(M)]])
# # w = (d1*(u@np.linalg.inv(C)) + d2*(M@np.linalg.inv(C)))/d3
# # sigma = w@C@np.transpose(w)
# # print(min_return)
# # print(w)

# d1 = np.linalg.det([[1,u@np.linalg.inv(C)@np.transpose(M)],[0.18,M@np.linalg.inv(C)@np.transpose(M)]])
# d2 = np.linalg.det([[u@np.linalg.inv(C)@np.transpose(u),1],[M@np.linalg.inv(C)@np.transpose(u),0.18]])
# d3 = np.linalg.det([[u@np.linalg.inv(C)@np.transpose(u),u@np.linalg.inv(C)@np.transpose(M)],[M@np.linalg.inv(C)@np.transpose(u),M@np.linalg.inv(C)@np.transpose(M)]])
# w = (d1*(u@np.linalg.inv(C)) + d2*(M@np.linalg.inv(C)))/d3
# sigma = w@C@np.transpose(w)
# print(math.sqrt(sigma))

# rf = 0.1
# u_dash = rf*u
# w = ((M-u_dash)@np.linalg.inv(C))/((M-u_dash)@np.linalg.inv(C)@np.transpose(u))
# print(w)

# return_market = M@np.transpose(w)
# risk_market = math.sqrt(w@C@np.transpose(w))

# cml_x = []
# cml_y = []

# for k in np.linspace(0,0.18,1000):
#     cml_x.append(k)
#     cml_y.append(rf + (return_market-rf)*k/risk_market)

# plt.plot(cml_x,cml_y,label='CML')
# plt.legend()
# plt.xlabel("Risk")
# plt.ylabel("Return")
# plt.show()

# risk_req = 0.1
# return_req = rf + (return_market-rf)*risk_req/risk_market
# w_rf = (return_req-return_market)/(rf-return_market)
# w_risky = 1-w_rf
# w_security = w*w_risky
# print(w_security)
# print(w_rf)

import numpy as np
import matplotlib.pyplot as plt
import math
from tabulate import tabulate
from scipy import interpolate

M = [0.1,0.2,0.15]
C = [[0.005,-0.010,0.004],[-0.010, 0.040, -0.002],[0.004, -0.002, 0.023]]
u = [1,1,1]
w_min = (u@np.linalg.inv(C))/(u@np.linalg.inv(C)@np.transpose(u))
sigma_min = math.sqrt(w_min@C@np.transpose(w_min))
u_min = M@np.transpose(w_min)

markovitz_x = []
markovitz_y = np.linspace(u_min,1.6*u_min,10000)
nonmarkovitz_x = []
nonmarkovitz_y = np.linspace(0.4*u_min,u_min,10000)

part_b = [['Serial Number','Weights','Return','Risk']]

idx = 0

for uv in markovitz_y:
    m1 = [[1,u@np.linalg.inv(C)@np.transpose(M)],[uv,M@np.linalg.inv(C)@np.transpose(M)]]
    m2 = [[u@np.linalg.inv(C)@np.transpose(u),1],[M@np.linalg.inv(C)@np.transpose(u),uv]]
    m3 = [[u@np.linalg.inv(C)@np.transpose(u),u@np.linalg.inv(C)@np.transpose(M)],[M@np.linalg.inv(C)@np.transpose(u),M@np.linalg.inv(C)@np.transpose(M)]]
    w = (np.linalg.det(m1)*(u@np.linalg.inv(C)) + np.linalg.det(m2)*(M@np.linalg.inv(C)))/np.linalg.det(m3)
    sigma = math.sqrt(w@C@np.transpose(w))
    markovitz_x.append(sigma)
    idx+=1
    if(idx%1000 == 0):
        part_b.append([idx/1000,w,uv,sigma])

plt.plot(markovitz_x,markovitz_y,color='red',label='Markovitz Frontier')

for uv in nonmarkovitz_y:
    m1 = [[1,u@np.linalg.inv(C)@np.transpose(M)],[uv,M@np.linalg.inv(C)@np.transpose(M)]]
    m2 = [[u@np.linalg.inv(C)@np.transpose(u),1],[M@np.linalg.inv(C)@np.transpose(u),uv]]
    m3 = [[u@np.linalg.inv(C)@np.transpose(u),u@np.linalg.inv(C)@np.transpose(M)],[M@np.linalg.inv(C)@np.transpose(u),M@np.linalg.inv(C)@np.transpose(M)]]
    w = (np.linalg.det(m1)*(u@np.linalg.inv(C)) + np.linalg.det(m2)*(M@np.linalg.inv(C)))/np.linalg.det(m3)
    sigma = math.sqrt(w@C@np.transpose(w))
    nonmarkovitz_x.append(sigma)
    


plt.plot(nonmarkovitz_x,nonmarkovitz_y,color='Black')
plt.plot(sigma_min,u_min,color='Green', label='Minimum Variance Portfolio',marker='o')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.grid(True)
plt.legend()
plt.show()

print("Minimum Risk = ", sigma_min)
print("Mean corresponding to Minimum Risk = ", u_min)

print("")

print("PART B-")
print(tabulate(part_b,headers='firstrow',tablefmt='grid'))

print("")

print("PART C-")
interpolate_func1 = interpolate.interp1d(markovitz_x,markovitz_y)
max_return = interpolate_func1(0.15)
c1 = [[1,u@np.linalg.inv(C)@np.transpose(M)],[max_return,M@np.linalg.inv(C)@np.transpose(M)]]
c2 = [[u@np.linalg.inv(C)@np.transpose(u),1],[M@np.linalg.inv(C)@np.transpose(u),max_return]]
c3 = [[u@np.linalg.inv(C)@np.transpose(u),u@np.linalg.inv(C)@np.transpose(M)],[M@np.linalg.inv(C)@np.transpose(u),M@np.linalg.inv(C)@np.transpose(M)]]
weight_C = (np.linalg.det(c1)*(u@np.linalg.inv(C)) + np.linalg.det(c2)*(M@np.linalg.inv(C)))/np.linalg.det(c3)
print("For risk = 15%,")
print("The maximum return is: ", max_return)
print("Corresponding weights of the portfolio are: ", weight_C)
print("")

interpolate_func2 = interpolate.interp1d(nonmarkovitz_x,nonmarkovitz_y)
min_return = interpolate_func2(0.15)
c1 = [[1,u@np.linalg.inv(C)@np.transpose(M)],[min_return,M@np.linalg.inv(C)@np.transpose(M)]]
c2 = [[u@np.linalg.inv(C)@np.transpose(u),1],[M@np.linalg.inv(C)@np.transpose(u),min_return]]
c3 = [[u@np.linalg.inv(C)@np.transpose(u),u@np.linalg.inv(C)@np.transpose(M)],[M@np.linalg.inv(C)@np.transpose(u),M@np.linalg.inv(C)@np.transpose(M)]]
weight_C = (np.linalg.det(c1)*(u@np.linalg.inv(C)) + np.linalg.det(c2)*(M@np.linalg.inv(C)))/np.linalg.det(c3)
print("For risk = 15%,")
print("The minimum return is: ", min_return)
print("Corresponding weights of the portfolio are: ", weight_C)


print("")

print("PART D-")
min_risk = np.interp(0.18,markovitz_y,markovitz_x)
print("For return = 18%, the minimum risk is: ", min_risk*100, " %")

print("")

print("PART E-")
u_rf = 0.10
urf_u = [0.1, 0.1, 0.1]
market_portfolio = np.subtract(M,urf_u)@np.linalg.inv(C)
gamma = 0
for i in market_portfolio:
    gamma += i
market_portfolio /= gamma
print("The market portfolio corresponding to risk free return 10%, is: ", market_portfolio,)
u_m = M@np.transpose(market_portfolio)
sigma_m = math.sqrt(market_portfolio@C@np.transpose(market_portfolio))
print("The Return is: ", u_m)
print("The Risk is: ", sigma_m*100, " %")

sigma_cml = np.linspace(0,0.2,1000)
u_cml = []
for sig in sigma_cml:
    u_cml.append(((u_m - u_rf)*sig)/sigma_m + u_rf)

plt.plot(sigma_cml,u_cml,color='Blue',label='Capital Market Line')
plt.plot(markovitz_x,markovitz_y,color='red',label='Markovitz Frontier')
plt.plot(nonmarkovitz_x,nonmarkovitz_y,color='Black')
plt.plot(sigma_m,u_m,color='Green', label='Market Portfolio',marker='o')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.grid(True)
plt.legend()
plt.show()

print("")

print("PART F-")
print("The required portfolio with risk = 10%, is: ")
sig = 0.1
u_req = ((u_m - u_rf)*sig)/sigma_m + u_rf
w_rf = (u_req - u_m)/(u_rf - u_m)
w_risky = 1-w_rf
w_security = market_portfolio*w_risky
print("Weight of risk-free security is: ", w_rf)
print("Weight of risky security is: ", w_security)
print("")

print("The required portfolio with risk = 25%, is: ")
sig = 0.25
u_req = ((u_m - u_rf)*sig)/sigma_m + u_rf
w_rf = (u_req - u_m)/(u_rf - u_m)
w_risky = 1-w_rf
w_security = market_portfolio*w_risky
print("Weight of risk-free security is: ", w_rf)
print("Weight of risky security is: ", w_security)
print("")
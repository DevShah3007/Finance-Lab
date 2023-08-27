# The Topic of this Assignment is Greeks and their senstivity analysis.

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# Define variables 
r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

def blackScholes(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        if type == "c":
            price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "p":
            price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        return price
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")

def delta_calc(r, S, K, T, sigma, type="c"):
    "Calculate delta of an option"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    try:
        if type == "c":
            delta_calc = norm.cdf(d1, 0, 1)
        elif type == "p":
            delta_calc = -norm.cdf(-d1, 0, 1)
        return delta_calc
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")

def gamma_calc(r, S, K, T, sigma, type="c"):
    "Calculate gamma of a option"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        gamma_calc = norm.pdf(d1, 0, 1)/(S*sigma*np.sqrt(T))
        return gamma_calc
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")

def vega_calc(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        vega_calc = S*norm.pdf(d1, 0, 1)*np.sqrt(T)
        return vega_calc*0.01
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")

def theta_calc(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        if type == "c":
            theta_calc = -S*norm.pdf(d1, 0, 1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "p":
            theta_calc = -S*norm.pdf(d1, 0, 1)*sigma/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        return theta_calc/365
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")

def rho_calc(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        if type == "c":
            rho_calc = K*T*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "p":
            rho_calc = -K*T*np.exp(-r*T)*norm.cdf(-d2, 0, 1)
        return rho_calc*0.01
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")

option_type='c'
print("For Call Option- ")
print("Option Price: ",  blackScholes(r, S, K, T, sigma, option_type))
print("       Delta: ",  delta_calc(r, S, K, T, sigma, option_type))
print("       Gamma: ",  gamma_calc(r, S, K, T, sigma, option_type))
print("       Vega : ",  vega_calc(r, S, K, T, sigma, option_type))
print("       Theta: ",  theta_calc(r, S, K, T, sigma, option_type))
print("       Rho  : ",  rho_calc(r, S, K, T, sigma, option_type))

print()

option_type='p'
print("For Put Option- ")
print("Option Price: ",  blackScholes(r, S, K, T, sigma, option_type))
print("       Delta: ",  delta_calc(r, S, K, T, sigma, option_type))
print("       Gamma: ",  gamma_calc(r, S, K, T, sigma, option_type))
print("       Vega : ",  vega_calc(r, S, K, T, sigma, option_type))
print("       Theta: ",  theta_calc(r, S, K, T, sigma, option_type))
print("       Rho  : ",  rho_calc(r, S, K, T, sigma, option_type))



## Sensitivity Analysis for Delta

##Delta vs Stock Price

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(10,60,500)
y_axis = []
for s in x_axis:
    y_axis.append(delta_calc(r, s, K, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Stock-Price (S)')
plt.ylabel('Delta')
plt.title('Stock-Price vs Delta for Call-Option')

x_axis = np.linspace(10,60,500)
y_axis = []
for s in x_axis:
    y_axis.append(delta_calc(r, s, K, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Stock-Price (S)')
plt.ylabel('Delta')
plt.title('Stock-Price vs Delta for Put-Option')
plt.show()


##Delta vs Strike Price

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(10,60,500)
y_axis = []
for k in x_axis:
    y_axis.append(delta_calc(r, S, k, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Strike-Price (K)')
plt.ylabel('Delta')
plt.title('Strike-Price vs Delta for Call-Option')

x_axis = np.linspace(10,60,500)
y_axis = []
for k in x_axis:
    y_axis.append(delta_calc(r, S, k, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Strike-Price (K)')
plt.ylabel('Delta')
plt.title('Strike-Price vs Delta for Put-Option')
plt.show()


##Delta vs Sigma

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(0.1,0.9,500)
y_axis = []
for sgm in x_axis:
    y_axis.append(delta_calc(r, S, K, T, sgm, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Sigma')
plt.ylabel('Delta')
plt.title('Sigma vs Delta for Call-Option')

x_axis = np.linspace(0.1,0.9,500)
y_axis = []
for sgm in x_axis:
    y_axis.append(delta_calc(r, S, K, T, sgm, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Sigma')
plt.ylabel('Delta')
plt.title('Sigma vs Delta for Put-Option')
plt.show()


##Delta vs Risk-Free Rate

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(0.01,0.1,500)
y_axis = []
for rr in x_axis:
    y_axis.append(delta_calc(rr, S, K, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Risk-Free Rate')
plt.ylabel('Delta')
plt.title('Risk-Free Rate vs Delta for Call-Option')

x_axis = np.linspace(0.01,0.1,500)
y_axis = []
for rr in x_axis:
    y_axis.append(delta_calc(rr, S, K, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Risk-Free Rate')
plt.ylabel('Delta')
plt.title('Risk-Free Rate vs Delta for Put-Option')
plt.show()



## Sensitivity Analysis for Gamma

##Gamma vs Stock Price

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(10,60,500)
y_axis = []
for s in x_axis:
    y_axis.append(gamma_calc(r, s, K, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Stock-Price (S)')
plt.ylabel('Gamma')
plt.title('Stock-Price vs Gamma for Call-Option')

x_axis = np.linspace(10,60,500)
y_axis = []
for s in x_axis:
    y_axis.append(gamma_calc(r, s, K, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Stock-Price (S)')
plt.ylabel('Gamma')
plt.title('Stock-Price vs Gamma for Put-Option')
plt.show()


##Gamma vs Strike Price

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(10,60,500)
y_axis = []
for k in x_axis:
    y_axis.append(gamma_calc(r, S, k, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Strike-Price (K)')
plt.ylabel('Gamma')
plt.title('Strike-Price vs Gamma for Call-Option')

x_axis = np.linspace(10,60,500)
y_axis = []
for k in x_axis:
    y_axis.append(gamma_calc(r, S, k, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Strike-Price (K)')
plt.ylabel('Gamma')
plt.title('Strike-Price vs Gamma for Put-Option')
plt.show()


##Gamma vs Sigma

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(0.1,0.9,500)
y_axis = []
for sgm in x_axis:
    y_axis.append(gamma_calc(r, S, K, T, sgm, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Sigma')
plt.ylabel('Gamma')
plt.title('Sigma vs Gamma for Call-Option')

x_axis = np.linspace(0.1,0.9,500)
y_axis = []
for sgm in x_axis:
    y_axis.append(gamma_calc(r, S, K, T, sgm, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Sigma')
plt.ylabel('Gamma')
plt.title('Sigma vs Gamma for Put-Option')
plt.show()


##Gamma vs Risk-Free Rate

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(0.01,0.1,500)
y_axis = []
for rr in x_axis:
    y_axis.append(gamma_calc(rr, S, K, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Risk-Free Rate')
plt.ylabel('Gamma')
plt.title('Risk-Free Rate vs Gamma for Call-Option')

x_axis = np.linspace(0.01,0.1,500)
y_axis = []
for rr in x_axis:
    y_axis.append(gamma_calc(rr, S, K, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Risk-Free Rate')
plt.ylabel('Gamma')
plt.title('Risk-Free Rate vs Gamma for Put-Option')
plt.show()



## Sensitivity Analysis for Vega

## 2-D plots

##Vega vs Stock Price

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(10,60,500)
y_axis = []
for s in x_axis:
    y_axis.append(vega_calc(r, s, K, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Stock-Price (S)')
plt.ylabel('Vega')
plt.title('Stock-Price vs Vega for Call-Option')

x_axis = np.linspace(10,60,500)
y_axis = []
for s in x_axis:
    y_axis.append(gamma_calc(r, s, K, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Stock-Price (S)')
plt.ylabel('Vega')
plt.title('Stock-Price vs Vega for Put-Option')
plt.show()


##Vega vs Strike Price

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(10,60,500)
y_axis = []
for k in x_axis:
    y_axis.append(vega_calc(r, S, k, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Strike-Price (K)')
plt.ylabel('Vega')
plt.title('Strike-Price vs Vega for Call-Option')

x_axis = np.linspace(10,60,500)
y_axis = []
for k in x_axis:
    y_axis.append(vega_calc(r, S, k, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Strike-Price (K)')
plt.ylabel('Vega')
plt.title('Strike-Price vs Vega for Put-Option')
plt.show()


##Vega vs Sigma

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(0.1,0.9,500)
y_axis = []
for sgm in x_axis:
    y_axis.append(vega_calc(r, S, K, T, sgm, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Sigma')
plt.ylabel('Vega')
plt.title('Sigma vs Vega for Call-Option')

x_axis = np.linspace(0.1,0.9,500)
y_axis = []
for sgm in x_axis:
    y_axis.append(vega_calc(r, S, K, T, sgm, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Sigma')
plt.ylabel('Vega')
plt.title('Sigma vs Vega for Put-Option')
plt.show()


##Vega vs Risk-Free Rate

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(0.01,0.1,500)
y_axis = []
for rr in x_axis:
    y_axis.append(vega_calc(rr, S, K, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Risk-Free Rate')
plt.ylabel('Vega')
plt.title('Risk-Free Rate vs Vega for Call-Option')

x_axis = np.linspace(0.01,0.1,500)
y_axis = []
for rr in x_axis:
    y_axis.append(vega_calc(rr, S, K, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Risk-Free Rate')
plt.ylabel('Vega')
plt.title('Risk-Free Rate vs Vega for Put-Option')
plt.show()



## Sensitivity Analysis for Theta

##Theta vs Stock Price

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(10,60,500)
y_axis = []
for s in x_axis:
    y_axis.append(theta_calc(r, s, K, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Stock-Price (S)')
plt.ylabel('Theta')
plt.title('Stock-Price vs Theta for Call-Option')

x_axis = np.linspace(10,60,500)
y_axis = []
for s in x_axis:
    y_axis.append(theta_calc(r, s, K, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Stock-Price (S)')
plt.ylabel('Theta')
plt.title('Stock-Price vs Theta for Put-Option')
plt.show()


##Theta vs Strike Price

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(10,60,500)
y_axis = []
for k in x_axis:
    y_axis.append(theta_calc(r, S, k, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Strike-Price (K)')
plt.ylabel('Theta')
plt.title('Strike-Price vs Theta for Call-Option')

x_axis = np.linspace(10,60,500)
y_axis = []
for k in x_axis:
    y_axis.append(theta_calc(r, S, k, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Strike-Price (K)')
plt.ylabel('Theta')
plt.title('Strike-Price vs Theta for Put-Option')
plt.show()


##Theta vs Sigma

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(0.1,0.9,500)
y_axis = []
for sgm in x_axis:
    y_axis.append(theta_calc(r, S, K, T, sgm, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Sigma')
plt.ylabel('Theta')
plt.title('Sigma vs Theta for Call-Option')

x_axis = np.linspace(0.1,0.9,500)
y_axis = []
for sgm in x_axis:
    y_axis.append(theta_calc(r, S, K, T, sgm, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Sigma')
plt.ylabel('Theta')
plt.title('Sigma vs Theta for Put-Option')
plt.show()


##Theta vs Risk-Free Rate

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(0.01,0.1,500)
y_axis = []
for rr in x_axis:
    y_axis.append(theta_calc(rr, S, K, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Risk-Free Rate')
plt.ylabel('Theta')
plt.title('Risk-Free Rate vs Theta for Call-Option')

x_axis = np.linspace(0.01,0.1,500)
y_axis = []
for rr in x_axis:
    y_axis.append(theta_calc(rr, S, K, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Risk-Free Rate')
plt.ylabel('Theta')
plt.title('Risk-Free Rate vs Theta for Put-Option')
plt.show()



## Sensitivity Analysis for Rho

##Rho vs Stock Price

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(10,60,500)
y_axis = []
for s in x_axis:
    y_axis.append(rho_calc(r, s, K, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Stock-Price (S)')
plt.ylabel('Rho')
plt.title('Stock-Price vs Rho for Call-Option')

x_axis = np.linspace(10,60,500)
y_axis = []
for s in x_axis:
    y_axis.append(rho_calc(r, s, K, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Stock-Price (S)')
plt.ylabel('Rho')
plt.title('Stock-Price vs Rho for Put-Option')
plt.show()


##Rho vs Strike Price

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(10,60,500)
y_axis = []
for k in x_axis:
    y_axis.append(rho_calc(r, S, k, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Strike-Price (K)')
plt.ylabel('Rho')
plt.title('Strike-Price vs Rho for Call-Option')

x_axis = np.linspace(10,60,500)
y_axis = []
for k in x_axis:
    y_axis.append(rho_calc(r, S, k, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Strike-Price (K)')
plt.ylabel('Rho')
plt.title('Strike-Price vs Rho for Put-Option')
plt.show()


##Rho vs Sigma

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(0.1,0.9,500)
y_axis = []
for sgm in x_axis:
    y_axis.append(rho_calc(r, S, K, T, sgm, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Sigma')
plt.ylabel('Rho')
plt.title('Sigma vs Rho for Call-Option')

x_axis = np.linspace(0.1,0.9,500)
y_axis = []
for sgm in x_axis:
    y_axis.append(rho_calc(r, S, K, T, sgm, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Sigma')
plt.ylabel('Rho')
plt.title('Sigma vs Rho for Put-Option')
plt.show()


##Rho vs Risk-Free Rate

r = 0.01
S = 30
K = 40
T = 240/365
sigma = 0.30

x_axis = np.linspace(0.01,0.1,500)
y_axis = []
for rr in x_axis:
    y_axis.append(rho_calc(rr, S, K, T, sigma, 'c'))
plt.subplot(1,2,1)
plt.plot(x_axis,y_axis)
plt.xlabel('Risk-Free Rate')
plt.ylabel('Rho')
plt.title('Risk-Free Rate vs Rho for Call-Option')

x_axis = np.linspace(0.01,0.1,500)
y_axis = []
for rr in x_axis:
    y_axis.append(rho_calc(rr, S, K, T, sigma, 'p'))
plt.subplot(1,2,2)
plt.plot(x_axis,y_axis)
plt.xlabel('Risk-Free Rate')
plt.ylabel('Rho')
plt.title('Risk-Free Rate vs Rho for Put-Option')
plt.show()


## 3-D plots

## Delta

S = 30
K = 40
sigma = 0.30
r = 0.01
T = 240.0/365

varying = []
varying.append(np.linspace(10,60,100))
varying.append(np.linspace(10,60,100))
varying.append(np.linspace(0.1,0.9,100))
varying.append(np.linspace(0.01,0.1,100))

names = ["S(t)", "Strike Price", "Sigma", "Risk-Free rate" ]

for i in range(4):
    for j in range(i+1,4):
        fig, ax = plt.subplots(nrows = 1,ncols = 2,subplot_kw={"projection": "3d"})
        call_p = np.zeros((100,100))
        put_p = np.zeros((100,100))
        p_axis, q_axis = np.meshgrid(varying[i],varying[j])
        params = [r,S,K,T,sigma]
        for ii in range(len(p_axis)):
            for jj in range(len(q_axis)):
                params[i]=varying[i][ii]
                params[j]=varying[j][jj]
                call_p[ii][jj]=delta_calc(params[0],params[1],params[2],params[3],params[4],'c')
                put_p[ii][jj]=delta_calc(params[0],params[1],params[2],params[3],params[4],'p')
        fig.suptitle("Delta vs "+names[i]+" vs "+names[j])
        fig.set_size_inches(12, 5)
        fig.set_dpi(150)
        ax[0].plot_surface(p_axis,q_axis,call_p,cmap='inferno')
        ax[0].set_title("Delta for Call-Option")
        ax[0].set_xlabel(names[i])
        ax[0].set_ylabel(names[j])
        ax[0].set_zlabel("Delta")
        ax[1].plot_surface(p_axis,q_axis,put_p,cmap='viridis')
        ax[1].set_title("Delta for Put-Option")
        ax[1].set_xlabel(names[i])
        ax[1].set_ylabel(names[j])
        ax[1].set_zlabel("Delta")
        ax[0].view_init(15,-135)
        ax[1].view_init(15,45)
        plt.show()


## Gamma

S = 30
K = 40
sigma = 0.30
r = 0.01
T = 240/365

varying = []
varying.append(np.linspace(10,60,100))
varying.append(np.linspace(10,60,100))
varying.append(np.linspace(0.1,0.9,100))
varying.append(np.linspace(0.01,0.1,100))

names = ["S(t)", "Strike Price", "Sigma", "Risk-Free rate" ]

for i in range(4):
    for j in range(i+1,4):
        fig, ax = plt.subplots(nrows = 1,ncols = 2,subplot_kw={"projection": "3d"})
        call_p = np.zeros((100,100))
        put_p = np.zeros((100,100))
        p_axis, q_axis = np.meshgrid(varying[i],varying[j])
        params = [r,S,K,T,sigma]
        for ii in range(len(p_axis)):
            for jj in range(len(q_axis)):
                params[i]=varying[i][ii]
                params[j]=varying[j][jj]
                call_p[ii][jj]=gamma_calc(params[0],params[1],params[2],params[3],params[4],'c')
                put_p[ii][jj]=gamma_calc(params[0],params[1],params[2],params[3],params[4],'p')
        fig.suptitle("Gamma vs "+names[i]+" vs "+names[j])
        fig.set_size_inches(12, 5)
        fig.set_dpi(150)
        ax[0].plot_surface(p_axis,q_axis,call_p,cmap='inferno')
        ax[0].set_title("Gamma for Call-Option")
        ax[0].set_xlabel(names[i])
        ax[0].set_ylabel(names[j])
        ax[0].set_zlabel("Gamma")
        ax[1].plot_surface(p_axis,q_axis,put_p,cmap='viridis')
        ax[1].set_title("Gamma for Put-Option")
        ax[1].set_xlabel(names[i])
        ax[1].set_ylabel(names[j])
        ax[1].set_zlabel("Gamma")
        ax[0].view_init(15,-135)
        ax[1].view_init(15,45)
        plt.show()


## Vega

S = 30
K = 40
sigma = 0.30
r = 0.01
T = 240/365

varying = []
varying.append(np.linspace(10,60,100))
varying.append(np.linspace(10,60,100))
varying.append(np.linspace(0.1,0.9,100))
varying.append(np.linspace(0.01,0.1,100))

names = ["S(t)", "Strike Price", "Sigma", "Risk-Free rate" ]

for i in range(4):
    for j in range(i+1,4):
        fig, ax = plt.subplots(nrows = 1,ncols = 2,subplot_kw={"projection": "3d"})
        call_p = np.zeros((100,100))
        put_p = np.zeros((100,100))
        p_axis, q_axis = np.meshgrid(varying[i],varying[j])
        params = [r,S,K,T,sigma]
        for ii in range(len(p_axis)):
            for jj in range(len(q_axis)):
                params[i]=varying[i][ii]
                params[j]=varying[j][jj]
                call_p[ii][jj]=vega_calc(params[0],params[1],params[2],params[3],params[4],'c')
                put_p[ii][jj]=vega_calc(params[0],params[1],params[2],params[3],params[4],'p')
        fig.suptitle("Vega vs "+names[i]+" vs "+names[j])
        fig.set_size_inches(12, 5)
        fig.set_dpi(150)
        ax[0].plot_surface(p_axis,q_axis,call_p,cmap='inferno')
        ax[0].set_title("Vega for Call-Option")
        ax[0].set_xlabel(names[i])
        ax[0].set_ylabel(names[j])
        ax[0].set_zlabel("Vega")
        ax[1].plot_surface(p_axis,q_axis,put_p,cmap='viridis')
        ax[1].set_title("Vega for Put-Option")
        ax[1].set_xlabel(names[i])
        ax[1].set_ylabel(names[j])
        ax[1].set_zlabel("Vega")
        ax[0].view_init(15,-135)
        ax[1].view_init(15,45)
        plt.show()


## Theta

S = 30
K = 40
sigma = 0.30
r = 0.01
T = 240/365

varying = []
varying.append(np.linspace(10,60,100))
varying.append(np.linspace(10,60,100))
varying.append(np.linspace(0.1,0.9,100))
varying.append(np.linspace(0.01,0.1,100))

names = ["S(t)", "Strike Price", "Sigma", "Risk-Free rate" ]

for i in range(4):
    for j in range(i+1,4):
        fig, ax = plt.subplots(nrows = 1,ncols = 2,subplot_kw={"projection": "3d"})
        call_p = np.zeros((100,100))
        put_p = np.zeros((100,100))
        p_axis, q_axis = np.meshgrid(varying[i],varying[j])
        params = [r,S,K,T,sigma]
        for ii in range(len(p_axis)):
            for jj in range(len(q_axis)):
                params[i]=varying[i][ii]
                params[j]=varying[j][jj]
                call_p[ii][jj]=theta_calc(params[0],params[1],params[2],params[3],params[4],'c')
                put_p[ii][jj]=theta_calc(params[0],params[1],params[2],params[3],params[4],'p')
        fig.suptitle("Theta vs "+names[i]+" vs "+names[j])
        fig.set_size_inches(12, 5)
        fig.set_dpi(150)
        ax[0].plot_surface(p_axis,q_axis,call_p,cmap='inferno')
        ax[0].set_title("Theta for Call-Option")
        ax[0].set_xlabel(names[i])
        ax[0].set_ylabel(names[j])
        ax[0].set_zlabel("Theta")
        ax[1].plot_surface(p_axis,q_axis,put_p,cmap='viridis')
        ax[1].set_title("Theta for Put-Option")
        ax[1].set_xlabel(names[i])
        ax[1].set_ylabel(names[j])
        ax[1].set_zlabel("Theta")
        ax[0].view_init(15,-135)
        ax[1].view_init(15,45)
        plt.show()

    
## Rho

S = 30
K = 40
sigma = 0.30
r = 0.01
T = 240/365

varying = []
varying.append(np.linspace(10,60,100))
varying.append(np.linspace(10,60,100))
varying.append(np.linspace(0.1,0.9,100))
varying.append(np.linspace(0.01,0.1,100))

names = ["S(t)", "Strike Price", "Sigma", "Risk-Free rate" ]

for i in range(4):
    for j in range(i+1,4):
        fig, ax = plt.subplots(nrows = 1,ncols = 2,subplot_kw={"projection": "3d"})
        call_p = np.zeros((100,100))
        put_p = np.zeros((100,100))
        p_axis, q_axis = np.meshgrid(varying[i],varying[j])
        params = [r,S,K,T,sigma]
        for ii in range(len(p_axis)):
            for jj in range(len(q_axis)):
                params[i]=varying[i][ii]
                params[j]=varying[j][jj]
                call_p[ii][jj]=rho_calc(params[0],params[1],params[2],params[3],params[4],'c')
                put_p[ii][jj]=rho_calc(params[0],params[1],params[2],params[3],params[4],'p')
        fig.suptitle("Rho vs "+names[i]+" vs "+names[j])
        fig.set_size_inches(12, 5)
        fig.set_dpi(150)
        ax[0].plot_surface(p_axis,q_axis,call_p,cmap='inferno')
        ax[0].set_title("Rho for Call-Option")
        ax[0].set_xlabel(names[i])
        ax[0].set_ylabel(names[j])
        ax[0].set_zlabel("Rho")
        ax[1].plot_surface(p_axis,q_axis,put_p,cmap='viridis')
        ax[1].set_title("Rho for Put-Option")
        ax[1].set_xlabel(names[i])
        ax[1].set_ylabel(names[j])
        ax[1].set_zlabel("Rho")
        ax[0].view_init(15,-135)
        ax[1].view_init(15,45)
        plt.show()
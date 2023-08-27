import numpy as np
import matplotlib.pyplot as plt

def GBM(S0, mu, sigma, n):
    dt = 1.0/252
    W = np.random.normal(0,1,n)
    S = []
    for i in range(n):
        s = S0*np.exp(sigma*W[i]*np.sqrt(dt) + (mu-0.5*sigma*sigma)*dt)
        S.append(s)
        S0 = s
    return S
        
def computePath(S0,mu,sigma,heading):
    n = 252
    x_axis = np.arange(252)
    for i in range(10):
        y_axis = GBM(S0,mu,sigma,n)
        plt.plot(x_axis,y_axis)
    plt.xlabel('Time (in days)')
    plt.ylabel('Stock Price')
    plt.title(heading)
    plt.show()


def asianOptionPrice(S0, K , T, mu, sigma):
    Call = []
    Put = []
    for i in range(100):
        S = GBM(S0,mu,sigma,252)
        call = max(np.mean(S)-K,0)
        put = max(K-np.mean(S),0)

        Call.append(np.exp(-mu*T)*call)
        Put.append(np.exp(-mu*T)*put)

    return np.mean(Call), np.mean(Put), np.var(Call), np.var(Put)

computePath(100,0.1,0.2,'Real World Measure')
computePath(100,0.05,0.2,'Risk Neutral Measure')



S0 = 100
r = 0.05
T = 126.0/252.0
sigma = 0.2
K = 105

for k in [90,105,110]:
    call_mean, put_mean, call_var, put_var = asianOptionPrice(S0, k, T, r,sigma)
    print("For K = ", k)
    print("Asian Call Option price: ", call_mean)
    print("Variance in Asian Call Option price: ", call_var)
    print()
    print("Asian Put Option price: ", put_mean)
    print("Variance in Asian Put Option price: ", put_var)
    print()
    print()



# Sensitivity Analysis

#S0
S0 = 100
r = 0.05
T = 126.0/252.0
sigma = 0.2
K = 105

x_axis = np.linspace(40,140,200)
call = []
put = []
for s in x_axis:
    call_mean, put_mean, call_var, put_var = asianOptionPrice(s, K, T, r,sigma)
    call.append(call_mean)
    put.append(put_mean)

plt.subplot(1,2,1)
plt.plot(x_axis,call)
plt.xlabel('Initial Stock Price (S0)')
plt.ylabel('Asian Call Option Price')
plt.title('Variation of Call Price with S0')
plt.subplot(1,2,2) 
plt.plot(x_axis,put)
plt.xlabel('Initial Stock Price (S0)')
plt.ylabel('Asian Put Option Price')
plt.title('Variation of Put Price with S0')
plt.show()


#K
S0 = 100
r = 0.05
T = 126.0/252.0
sigma = 0.2
K = 105

x_axis = np.linspace(40,140,200)
call = []
put = []
for k in x_axis:
    call_mean, put_mean, call_var, put_var = asianOptionPrice(S0, k, T, r,sigma)
    call.append(call_mean)
    put.append(put_mean)

plt.subplot(1,2,1)
plt.plot(x_axis,call)
plt.xlabel('Strike Price (K)')
plt.ylabel('Asian Call Option Price')
plt.title('Variation of Call Price with K')
plt.subplot(1,2,2)
plt.plot(x_axis,put)
plt.xlabel('Strike Price (K)')
plt.ylabel('Asian Put Option Price')
plt.title('Variation of Put Price with K')
plt.show()


#r
S0 = 100
r = 0.05
T = 126.0/252.0
sigma = 0.2
K = 105

x_axis = np.linspace(0.0,0.5,200)
call = []
put = []
for rr in x_axis:
    call_mean, put_mean, call_var, put_var = asianOptionPrice(S0, K, T, rr,sigma)
    call.append(call_mean)
    put.append(put_mean)

plt.subplot(1,2,1)
plt.plot(x_axis,call)
plt.xlabel('Risk Free Rate (r)')
plt.ylabel('Asian Call Option Price')
plt.title('Variation of Call Price with r')
plt.subplot(1,2,2)
plt.plot(x_axis,put)
plt.xlabel('Risk Free Rate (r)')
plt.ylabel('Asian Put Option Price')
plt.title('Variation of Put Price with r')
plt.show()


#sigma
S0 = 100
r = 0.05
T = 126.0/252.0
sigma = 0.2
K = 105

x_axis = np.linspace(0.0,1.0,200)
call = []
put = []
for sig in x_axis:
    call_mean, put_mean, call_var, put_var = asianOptionPrice(S0, K, T, r,sig)
    call.append(call_mean)
    put.append(put_mean)

plt.subplot(1,2,1)
plt.plot(x_axis,call)
plt.xlabel('Volatility (sigma)')
plt.ylabel('Asian Call Option Price')
plt.title('Variation of Call Price with sigma')
plt.subplot(1,2,2)
plt.plot(x_axis,put)
plt.xlabel('Volatility (sigma)')
plt.ylabel('Asian Put Option Price')
plt.title('Variation of Put Price with sigma')
plt.show()



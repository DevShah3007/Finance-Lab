import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def GBM(S0, mu, sigma, n):
    dt = 1.0/252
    W = np.random.normal(0,1,n)
    S = []
    for i in range(n):
        s = S0*np.exp(sigma*W[i]*np.sqrt(dt) + (mu-0.5*sigma*sigma)*dt)
        S.append(s)
        S0 = s
    return S
        
def variance_reduction(option_payoff, control_variate, r, T):
  X_bar = np.mean(control_variate)
  Y_bar = np.mean(option_payoff)

  max_iter = len(option_payoff)
  num, denom = 0, 0

  for idx in range(max_iter):
    num += (control_variate[idx] - X_bar) * (option_payoff[idx] - Y_bar)
    denom += (control_variate[idx] - X_bar) * (control_variate[idx] - X_bar)

  b = num/denom
  reduced_variate = []
  for idx in range(max_iter):
    reduced_variate.append((option_payoff[idx] - b*(control_variate[idx] - X_bar) * np.exp(-r*T)))
  
  return reduced_variate


def asianOptionPrice(S0, K , T, mu, sigma):
    Call = []
    Put = []
    cv_Call = []
    cv_Put = []
    for i in range(100):
        S = GBM(S0,mu,sigma,252)
        call = max(np.mean(S)-K,0)
        put = max(K-np.mean(S),0)

        Call.append(np.exp(-mu*T)*call)
        Put.append(np.exp(-mu*T)*put)

        cv_Call.append(np.exp(-mu*T)*max(K-S[len(S)-1],0))
        cv_Put.append(np.exp(-mu*T)*max(S[len(S)-1]-K,0))

    Call = variance_reduction(Call,cv_Call,mu,T)
    Put = variance_reduction(Put,cv_Put,mu,T)

    return np.mean(Call), np.mean(Put), np.var(Call), np.var(Put)



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



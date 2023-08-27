import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def compute_d1(sigma,T,t,St,K,r):
    if t >= T:
        return 0
    return (math.log(St/K) + (r+(sigma*sigma)/2.0)*(T-t))/(sigma*math.sqrt(T-t))

def compute_d2(sigma,T,t,St,K,r):
    if t >= T:
        return 0
    return (math.log(St/K) + (r-(sigma*sigma)/2.0)*(T-t))/(sigma*math.sqrt(T-t))

def compute_c(sigma,T,t,St,K,r):
    if t >= T:
        return max(0,St-K)
    d1 = compute_d1(sigma,T,t,St,K,r)
    d2 = compute_d2(sigma,T,t,St,K,r)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    return St*N_d1 - K*math.e**(-r*(T-t))*N_d2

def compute_p(sigma,T,t,St,K,r):
    if t >= T:
        return max(0,K-St)
    return compute_c(sigma,T,t,St,K,r) - St + K*math.e**(-r*(T-t))

T = 1.0
K = 1.0
r = 0.05
sigma = 0.6
t = 0
St = 1

print("Call price for t = ", t, " and S = ", St, " is ", compute_c(sigma,T,t,St,K,r) )
print("Put price for t = ", t, " and S = ", St, " is ", compute_p(sigma,T,t,St,K,r) )


import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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

for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    S = np.linspace(0.4,1.4,100)
    C = []
    T = 1.0
    K = 1.0
    r = 0.05
    sigma = 0.6
    for s in S:
        C.append(compute_c(sigma,T,t,s,K,r))
        strng = "t = " + str(t)
    plt.plot(S,C,label=strng)
    plt.legend()

plt.title("Call Option Price")
plt.xlabel("s")
plt.ylabel("C(t,s)")
plt.show()

for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    S = np.linspace(0.4,1.4,100)
    P = []
    T = 1.0
    K = 1.0
    r = 0.05
    sigma = 0.6
    for s in S:
        P.append(compute_p(sigma,T,t,s,K,r))
    strng = "t = " + str(t)
    plt.plot(S,P,label=strng)
    plt.legend()

plt.title("Put Option Price")
plt.xlabel("s")
plt.ylabel("P(t,s)")
plt.show()



T = 1.0
K = 1.0
r = 0.05
sigma = 0.6

ss = np.linspace(0.4,1.4,100)

fig = plt.figure()
ax = plt.axes(projection='3d')
tt = []
C = []
S = []
for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    for s in ss:
        C.append(compute_c(sigma,T,t,s,K,r))
        tt.append(t)
        S.append(s)
ax.scatter3D(tt,S,C,cmap='Greens')
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price")
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
tt = []
P = []
S = []
for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    for s in ss:
        P.append(compute_p(sigma,T,t,s,K,r))
        tt.append(t)
        S.append(s)

ax.scatter3D(tt,S,P,cmap='Greens')
ax.set_xlabel("t")
ax.set_ylabel("s")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price")
plt.show()
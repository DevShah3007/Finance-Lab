import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
cm = plt.get_cmap("RdYlGn")
col = [cm(float(i)/(10000)) for i in range(10000)]

def compute_d1(sigma,T,t,St,K,r):
    if t == T:
        return 0
    return (math.log(St/K) + (r+(sigma*sigma)/2.0)*(T-t))/(sigma*math.sqrt(T-t))

def compute_d2(sigma,T,t,St,K,r):
    return (math.log(St/K) + (r-(sigma*sigma)/2.0)*(T-t))/(sigma*math.sqrt(T-t))

def compute_c(sigma,T,t,St,K,r):
    if t == T:
        return max(0,St-K)
    d1 = compute_d1(sigma,T,t,St,K,r)
    d2 = compute_d2(sigma,T,t,St,K,r)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    return St*N_d1 - K*math.e**(-r*(T-t))*N_d2

def compute_p(sigma,T,t,St,K,r):
    if t == T:
        return max(0,K-St)
    return compute_c(sigma,T,t,St,K,r) - St + K*math.e**(-r*(T-t))

T = 1.0
K = 1.0
r = 0.05
sigma = 0.6

x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(sigma,T,i,j,K,r))
        P.append(compute_p(sigma,T,i,j,K,r))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=100)
ax.set_xlabel("t")
ax.set_ylabel("S(t)")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs t vs S(t)")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=100)
ax.set_xlabel("t")
ax.set_ylabel("S(t)")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs t vs S(t)")
plt.show()



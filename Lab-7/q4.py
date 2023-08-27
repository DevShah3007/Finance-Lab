import math
from scipy.stats import norm
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
cm = plt.get_cmap("RdYlGn")
col = [cm(float(i)/(10000)) for i in range(10000)]

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

#------- 2-D PLOTS -------

#-------------------------------------------
# For T
T = 1.0
K = 1.0
r = 0.05
sigma = 0.6
St = 0.8

for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    temp = np.linspace(0.1,1.0,100)
    C = []
    for i in temp:
        C.append(compute_c(sigma,i,t,St,K,r))
        strng = "t = " + str(t)
    plt.plot(temp,C,label=strng)
    plt.legend()

plt.title("Call Option Price vs T")
plt.xlabel("T")
plt.ylabel("C(t,s)")
plt.show()

for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    temp = np.linspace(0.1,1.0,100)
    P = []
    for i in temp:
        P.append(compute_p(sigma,i,t,St,K,r))
    strng = "t = " + str(t)
    plt.plot(temp,P,label=strng)
    plt.legend()

plt.title("Put Option Price vs T")
plt.xlabel("T")
plt.ylabel("P(t,s)")
plt.show()


#-------------------------------------------
# For r
T = 1.0
K = 1.0
r = 0.05
sigma = 0.6
St = 0.8

for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    temp = np.linspace(0.1,1.0,100)
    C = []
    for i in temp:
        C.append(compute_c(sigma,T,t,St,K,i))
        strng = "t = " + str(t)
    plt.plot(temp,C,label=strng)
    plt.legend()

plt.title("Call Option Price vs r")
plt.xlabel("r")
plt.ylabel("C(t,s)")
plt.show()

for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    temp = np.linspace(0.1,1.0,100)
    P = []
    for i in temp:
        P.append(compute_p(sigma,T,t,St,K,i))
    strng = "t = " + str(t)
    plt.plot(temp,P,label=strng)
    plt.legend()

plt.title("Put Option Price vs r")
plt.xlabel("r")
plt.ylabel("P(t,s)")
plt.show()


#-------------------------------------------
# For sigma
T = 1.0
K = 1.0
r = 0.05
sigma = 0.6
St = 0.8

for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    temp = np.linspace(0.1,1.0,100)
    C = []
    for i in temp:
        C.append(compute_c(i,T,t,St,K,r))
        strng = "t = " + str(t)
    plt.plot(temp,C,label=strng)
    plt.legend()

plt.title("Call Option Price vs sigma")
plt.xlabel("sigma")
plt.ylabel("C(t,s)")
plt.show()

for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    temp = np.linspace(0.1,1.0,100)
    P = []
    for i in temp:
        P.append(compute_p(i,T,t,St,K,r))
    strng = "t = " + str(t)
    plt.plot(temp,P,label=strng)
    plt.legend()

plt.title("Put Option Price vs sigma")
plt.xlabel("sigma")
plt.ylabel("P(t,s)")
plt.show()


#-------------------------------------------
# For K
T = 1.0
K = 1.0
r = 0.05
sigma = 0.6
St = 0.8

for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    temp = np.linspace(0.1,1.0,100)
    C = []
    for i in temp:
        C.append(compute_c(sigma,T,t,St,i,r))
        strng = "t = " + str(t)
    plt.plot(temp,C,label=strng)
    plt.legend()

plt.title("Call Option Price vs K")
plt.xlabel("K")
plt.ylabel("C(t,s)")
plt.show()

for t in [0.0,0.2,0.4,0.6,0.8,1.0]:
    temp = np.linspace(0.1,1.0,100)
    P = []
    for i in temp:
        P.append(compute_p(sigma,T,t,St,i,r))
    strng = "t = " + str(t)
    plt.plot(temp,P,label=strng)
    plt.legend()

plt.title("Put Option Price vs K")
plt.xlabel("K")
plt.ylabel("P(t,s)")
plt.show()


#------- 3-D PLOTS -------
T = 1.0
K = 1.0
r = 0.05
sigma = 0.6
St = 0.8
t = 0

#-------------------------------------------
# For T and K
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(sigma,i,t,St,j,r))
        P.append(compute_p(sigma,i,t,St,j,r))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("T")
ax.set_ylabel("K")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs T vs K")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("T")
ax.set_ylabel("K")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs T vs K")
plt.show()

#-------------------------------------------
# For T and r
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(sigma,i,t,St,K,j))
        P.append(compute_p(sigma,i,t,St,K,j))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("T")
ax.set_ylabel("r")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs T vs r")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("T")
ax.set_ylabel("r")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs T vs r")
plt.show()

#-------------------------------------------
# For T and sigma
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(j,i,t,St,K,r))
        P.append(compute_p(j,i,t,St,K,r))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("T")
ax.set_ylabel("sigma")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs T vs sigma")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("T")
ax.set_ylabel("sigma")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs T vs sigma")
plt.show()

#-------------------------------------------
# For T and S(t)
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(sigma,i,t,j,K,r))
        P.append(compute_p(sigma,i,t,j,K,r))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("T")
ax.set_ylabel("S(t)")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs T vs S(t)")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("T")
ax.set_ylabel("S(t)")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs T vs S(t)")
plt.show()

#-------------------------------------------
# For T and t
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(sigma,i,j,St,K,r))
        P.append(compute_p(sigma,i,j,St,K,r))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("T")
ax.set_ylabel("t")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs T vs t")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("T")
ax.set_ylabel("t")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs T vs t")
plt.show()

#-------------------------------------------
# For K and r
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(sigma,T,t,St,i,j))
        P.append(compute_p(sigma,T,t,St,i,j))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("K")
ax.set_ylabel("r")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs K vs r")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("K")
ax.set_ylabel("r")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs K vs r")
plt.show()

#-------------------------------------------
# For K and sigma
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(j,T,t,St,i,r))
        P.append(compute_p(j,T,t,St,i,r))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("K")
ax.set_ylabel("sigma")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs K vs sigma")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("K")
ax.set_ylabel("sigma")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs K vs sigma")
plt.show()

#-------------------------------------------
# For K and S(t)
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(sigma,T,t,j,i,r))
        P.append(compute_p(sigma,T,t,j,i,r))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("K")
ax.set_ylabel("S(t)")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs K vs S(t)")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("K")
ax.set_ylabel("S(t)")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs K vs S(t)")
plt.show()

#-------------------------------------------
# For K and t
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(sigma,T,j,St,i,r))
        P.append(compute_p(sigma,T,j,St,i,r))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("K")
ax.set_ylabel("t")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs K vs t")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("K")
ax.set_ylabel("t")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs K vs t")
plt.show()

#-------------------------------------------
# For r and sigma
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(j,T,t,St,K,i))
        P.append(compute_p(j,T,t,St,K,i))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("r")
ax.set_ylabel("sigma")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs r vs sigma")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("r")
ax.set_ylabel("sigma")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs r vs sigma")
plt.show()

#-------------------------------------------
# For r and S(t)
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(sigma,T,t,j,K,i))
        P.append(compute_p(sigma,T,t,j,K,i))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("r")
ax.set_ylabel("S(t)")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs r vs S(t)")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("r")
ax.set_ylabel("S(t)")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs r vs S(t)")
plt.show()

#-------------------------------------------
# For r and t
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(sigma,T,j,St,K,i))
        P.append(compute_p(sigma,T,t,St,K,i))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("r")
ax.set_ylabel("t")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs r vs t")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("r")
ax.set_ylabel("t")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs r vs t")
plt.show()

#-------------------------------------------
# For sigma and S(t)
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(i,T,t,j,K,r))
        P.append(compute_p(i,T,t,j,K,r))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("sigma")
ax.set_ylabel("S(t)")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs sigma vs S(t)")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("sigma")
ax.set_ylabel("S(t)")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs sigma vs S(t)")
plt.show()

#-------------------------------------------
# For sigma and t
x = np.linspace(0.1,1.0,100)
y = np.linspace(0.1,1.0,100)
C = []
P = []
xx = []
yy = []

for j in y:
    for i in x:
        C.append(compute_c(i,T,j,St,K,r))
        P.append(compute_p(i,T,j,St,K,r))
        xx.append(i)
        yy.append(j)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,C,c=col,s=10)
ax.set_xlabel("sigma")
ax.set_ylabel("t")
ax.set_zlabel("C(t,s)")
ax.set_title("Call Option Price vs sigma vs t")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(xx,yy,P,c=col,s=10)
ax.set_xlabel("sigma")
ax.set_ylabel("t")
ax.set_zlabel("P(t,s)")
ax.set_title("Put Option Price vs sigma vs t")
plt.show()
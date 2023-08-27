import math,numpy as np
import matplotlib.pyplot as plt

s=100
k=100
T=1
M=100
r=0.08
sigma=0.2
X=np.arange(0,200,1)
Xr=np.arange(0,1,0.01)
XM=np.arange(50,201,1)

def plot2d(x,y,xlabel,ylabel,title):
    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def call(s,x,sigma,r,T,M):
    dt=T/M
    R=math.exp(r*dt)
    u=math.exp((sigma*(math.sqrt(dt)))+(r-0.5*sigma*sigma)*dt)
    d=math.exp(-1*(sigma*(math.sqrt(dt)))+(r-0.5*sigma*sigma)*dt)
    if not (d<R and R<u):
        return
    p=(R-d)/(u-d)
    price=np.zeros((M+1,M+1))
    for i in range(M+1):
        price[M][i]=max(0,(s*(math.pow(u,M-i))*(math.pow(d,i)))-x)
    for j in range(M-1,-1,-1):
        for i in range(j+1):
            pp=s*(math.pow(u,j-i))*(math.pow(d,i))
            pp=max(pp-x,0)
            price[j][i]=max(((p*price[j+1][i])+((1-p)*price[j+1][i+1]))/R,pp)
    return price[0][0]


def put(s,x,sigma,r,T,M):
    dt=T/M
    R=math.exp(r*dt)
    u=math.exp((sigma*(math.sqrt(dt)))+(r-0.5*sigma*sigma)*dt)
    d=math.exp(-1*(sigma*(math.sqrt(dt)))+(r-0.5*sigma*sigma)*dt)
    if not (d<R and R<u):
        return
    p=(R-d)/(u-d)
    price=np.zeros((M+1,M+1))
    for i in range(M+1):
        price[M][i]=max(0,x-(s*(math.pow(u,M-i))*(math.pow(d,i))))
    for j in range(M-1,-1,-1):
        for i in range(j+1):
            pp=s*(math.pow(u,j-i))*(math.pow(d,i))
            pp=max(x-pp,0)
            price[j][i]=max(pp,((p*price[j+1][i])+((1-p)*price[j+1][i+1]))/R)
    return price[0][0]


print("The initial price of the call option is", call(s,k,sigma,r,T,M))
print("The initial price of the put option is", put(s,k,sigma,r,T,M))

#varry S0

s2call=[]
s2put=[]

for i in range(200):
    s2call.append(call(i,k,sigma,r,T,M))
    s2put.append(put(i,k,sigma,r,T,M))

plot2d(X,s2call,"S(0)","Prices of Call option at t=0","Initial Call option price vs S0 ")
plot2d(X,s2put,"S(0)","Prices of put option at t=0","Initial put option price vs S0 ")

#varry k

k2call=[]
k2put=[]

for i in range(200):
    k2call.append(call(s,i,sigma,r,T,M))
    k2put.append(put(s,i,sigma,r,T,M))

plot2d(X,k2call,"K","Prices of Call option at t=0","Initial Call option price vs K ")
plot2d(X,k2put,"K","Prices of put option at t=0","Initial put option price vs K ")

#varry r

r2call=[]
r2put=[]

for i in Xr:
    r2call.append(call(s,k,sigma,i,T,M))
    r2put.append(put(s,k,sigma,i,T,M))

plot2d(Xr,r2call,"r","Prices of Call option at t=0","Initial Call option price vs r ")
plot2d(Xr,r2put,"r","Prices of put option at t=0","Initial put option price vs r ")

#varry sigma


sigma2call=[]
sigma2put=[]

for i in Xr:
    sigma2call.append(call(s,k,i,r,T,M))
    sigma2put.append(put(s,k,i,r,T,M))

plot2d(Xr,sigma2call,"sigma","Prices of Call option at t=0","Initial Call option price vs sigma ")
plot2d(Xr,sigma2put,"sigma","Prices of put option at t=0","Initial put option price vs sigma ")

#varry M for k=95

k=95
M2call=[]
M2put=[]

for i in XM:
    M2call.append(call(s,k,sigma,r,T,i))
    M2put.append(put(s,k,sigma,r,T,i))

plot2d(XM,M2call,"M","Prices of Call option at t=0","Initial Call option price vs M for K=95")
plot2d(XM,M2put,"M","Prices of put option at t=0","Initial put option price vs M for K=95")

#varry M for k=100

k=100
M2call=[]
M2put=[]

for i in XM:
    M2call.append(call(s,k,sigma,r,T,i))
    M2put.append(put(s,k,sigma,r,T,i))

plot2d(XM,M2call,"M","Prices of Call option at t=0","Initial Call option price vs M for K=100")
plot2d(XM,M2put,"M","Prices of put option at t=0","Initial put option price vs M for K=100")

#varry M for k=105

k=105
M2call=[]
M2put=[]

for i in XM:
    M2call.append(call(s,k,sigma,r,T,i))
    M2put.append(put(s,k,sigma,r,T,i))

plot2d(XM,M2call,"M","Prices of Call option at t=0","Initial Call option price vs M for K=105")
plot2d(XM,M2put,"M","Prices of put option at t=0","Initial put option price vs M for K=105")


import numpy as np
import matplotlib.pyplot as plt

def computeA(a,b,sigma,T):
    if(T==0):
        return 0
    return ((a*b-0.5*sigma*sigma)*(computeB(a,b,sigma,T)-T))/(a*a) - (sigma*sigma*computeB(a,b,sigma,T)*computeB(a,b,sigma,T))/(4*a)

def computeB(a,b,sigma,T):
    if(T==0):
        return 0
    return (1/a)*(1-np.exp(-a*T))

def vasicekModel(beta,mu,sigma,r):
    a = beta
    b = beta*mu
    x_axis = [0]
    y_axis = [r]
    for t in range(1,500):
        x_axis.append(t)
        B = computeB(a,b,sigma,t)
        A = computeA(a,b,sigma,t)
        y_axis.append((B*r-A)/t)
    plt.plot(x_axis,y_axis)

# vasicekModel(5.9,0.2,0.3,0.1)
# vasicekModel(3.9,0.1,0.3,0.2)
# vasicekModel(0.1,0.4,0.11,0.1)
# plt.show()

for r in np.linspace(0,1,10):
    vasicekModel(0.1,0.4,0.11,r)

plt.show()


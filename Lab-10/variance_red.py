import numpy as np
import matplotlib.pyplot as plt

def GBM(S0,drift,sigma):
    dt = 1.0/252
    W = np.random.normal(0,1,252)
    S = []
    S_red = []
    S.append(S0)
    S_red.append(S0)
    for i in range(1,252):
        S.append(S[i-1]*np.exp(sigma*np.sqrt(dt)*W[i-1]+(drift-0.5*sigma*sigma)*dt))
        S_red.append(S_red[i-1]*np.exp(-sigma*np.sqrt(dt)*W[i-1]+(drift-0.5*sigma*sigma)*dt))
    return S, S_red

def computePath(S0,mu,sigma):
    x_axis = np.arange(252)
    for i in range(10):
        y_axis,y2_axis = GBM(S0,mu,sigma)
        plt.plot(x_axis,y_axis,c='red')
        plt.plot(x_axis,y2_axis,c='blue')
    plt.xlabel('Time (in days)')
    plt.ylabel('Stock Price')
    plt.show()


computePath(100,0.1,0.2)
computePath(100,0.05,0.2)

def asianOption(S0,drift,sigma,T, K):
    Call = []
    Put = []
    Call_red = []
    Put_red = []
    for i in range(100):
        S,S_red = GBM(S0,drift,sigma)
        call = max(0,np.mean(S)-K)
        put = max(0,K-np.mean(S))
        call_red = max(0,np.mean(S_red)-K)
        put_red = max(0,K-np.mean(S_red))

        Call.append(np.exp(-drift*T)*call)
        Call_red.append(np.exp(-drift*T)*0.5*(call+call_red))
        Put.append(np.exp(-drift*T)*put)
        Put_red.append(np.exp(-drift*T)*0.5*(put+put_red))

    print(np.mean(Call))
    print(np.mean(Put))
    print()
    print(np.var(Call))
    print(np.var(Call_red))
    print()
    print(np.var(Put))
    print(np.var(Put_red))

asianOption(100,0.05,0.2,0.5,110)



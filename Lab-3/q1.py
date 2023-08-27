import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.style.use("seaborn")


def binomial_model(S0, K, r, sigma, M):
    # initialise asset prices at maturity - Time step N
    M = int(M)
    dt = T / M
    u = np.exp(sigma * np.sqrt(dt) + (r - (sigma**2) / 2) * dt)
    d = np.exp(-sigma * np.sqrt(dt) + (r - (sigma**2) / 2) * dt)
    R = np.exp(r*dt)
    p = (R-d)/(u-d)


    S = np.zeros(M + 1)
    S[0] = S0 * (d**M)
    for j in range(1, M + 1):
        S[j] = S[j - 1] * u / d

    # initialise option value at maturity
    C = np.zeros(M + 1)
    P = np.zeros(M + 1)
    for j in range(0, M + 1):
        C[j] = max(0, S[j] - K)
        P[j] = max(0, K - S[j])

    # step backwards through tree
    for i in np.arange(M-1, -1, -1):
        for j in range(i+1):
            C[j] = max((p * C[j + 1] + (1 - p) * C[j])/R, S0*(d**i)*((1.0*u/d)**j) - K)
            P[j] = max((p * P[j + 1] + (1 - p) * P[j])/R, K - S0*(d**i)*((1.0*u/d)**j))
    
    return C[0],P[0]


def plot_2d():
    for i in range(4):
        fig1, ax1 = plt.subplots()
        ax1.plot(A[i], call2d[i][0], label='Call Prices')
        ax1.plot(A[i], put2d[i][0], label='Put Prices')
        ax1.legend()
        ax1.set_title(f'Option Price vs {x_label[i]} (Set 1 of u,d)')
        ax1.set_xlabel(f'{x_label[i]}')
        ax1.set_ylabel('Option Price')


# Initialise parameters
S0 = 100  # initial stock price
K = 100  # strike price
T = 1  # time to maturity in years
r = 0.08  # annual risk free rate
sigma = 0.2
M = 100

print(binomial_model(100,100,0.08,0.2,100))

# Senstivity analysis
left = [50, 50, 0.06, 0.05]
right = [150, 150, 0.7, 0.35]
N = 30
x_label = ['Initial Stock Price', 'Strike Price', 'Risk Free Rate', 'Volatility']
A = np.zeros((4, N))
for i in range(4):
    A[i] = np.linspace(left[i], right[i], N)

call2d = np.zeros((4, N))
put2d = np.zeros((4, N))

for j in np.arange(N):
    call2d[0][j], put2d[0][j] = binomial_model(A[0][j], K, r, sigma, M)
    call2d[1][j], put2d[1][j] = binomial_model(S0, A[1][j], r, sigma, M)
    call2d[2][j], put2d[2][j] = binomial_model(S0, K, A[2][j], sigma, M)
    call2d[3][j], put2d[3][j] = binomial_model(S0, K, r, A[3][j], M)


def parte():
    call_e = np.zeros((3, N))
    put_e = np.zeros((3, N))
    strike_price = [95,100,105]
    B = np.zeros((3,N))
    for k in len(strike_price):
        B[k] = np.linspace(strike_price[k]-50,strike_price[k]+50,N)
        for j in np.arange(N):
            call_e[0][j], put_e[0][j] = binomial_model(S0, strike_price[k], r, sigma, B[k][j])
        fig1, ax1 = plt.subplots()
        ax1.plot(B[k], call_e[0], label='Call Prices')
        ax1.plot(B[k], put_e[0], label='Put Prices')
        ax1.legend()
        ax1.set_title(f'Option Price vs {x_label[4]} (Set 1 of u,d) K={strike_price[k]}')
        ax1.set_xlabel(f'{x_label[4]}')
        ax1.set_ylabel('Option Price')
        fig1.savefig(f'fig1{4}e.png')


plot_2d()
parte()
plt.tight_layout()
plt.show()

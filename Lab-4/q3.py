import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import io

def compute_weights(M, C, mu):
  C_inverse = np.linalg.inv(C)
  u = [1 for i in range(len(M))]

  p = [[1, u @ C_inverse @ np.transpose(M)], [mu, M @ C_inverse @ np.transpose(M)]]
  q = [[u @ C_inverse @ np.transpose(u), 1], [M @ C_inverse @ np.transpose(u), mu]]
  r = [[u @ C_inverse @ np.transpose(u), u @ C_inverse @ np.transpose(M)], [M @ C_inverse @ np.transpose(u), M @ C_inverse @ np.transpose(M)]]

  det_p, det_q, det_r = np.linalg.det(p), np.linalg.det(q), np.linalg.det(r)
  det_p /= det_r
  det_q /= det_r

  w = det_p * (u @ C_inverse) + det_q * (M @ C_inverse)
  
  return w


def main():
    df = pd.read_csv("./Stock.csv")
    df.set_index('Date', inplace=True)
    df = df.pct_change()
    M = np.mean(df, axis = 0) * 12
    C = df.cov()

    returns = np.linspace(-2, 2, num = 5000)
    u = np.array([1 for i in range(len(M))])
    risk = []

    for mu in returns:
        w = compute_weights(M, C, mu)
        sigma = math.sqrt(w @ C @ np.transpose(w))
        risk.append(sigma)
    
    weight_min_var = u @ np.linalg.inv(C) / (u @ np.linalg.inv(C) @ np.transpose(u))
    mu_min_var = weight_min_var @ np.transpose(M)
    risk_min_var = math.sqrt(weight_min_var @ C @ np.transpose(weight_min_var))

    returns_plot1, risk_plot1, returns_plot2, risk_plot2 = [], [], [], []
    for i in range(len(returns)):
        if returns[i] >= mu_min_var: 
            returns_plot1.append(returns[i])
            risk_plot1.append(risk[i])
        else:
            returns_plot2.append(returns[i])
            risk_plot2.append(risk[i])


    mu_rf = 0.05

    market_portfolio_weights = (M - mu_rf * u) @ np.linalg.inv(C) / ((M - mu_rf * u) @ np.linalg.inv(C) @ np.transpose(u) )
    mu_market = market_portfolio_weights @ np.transpose(M)
    risk_market = math.sqrt(market_portfolio_weights @ C @ np.transpose(market_portfolio_weights))

    plt.plot(risk_plot1, returns_plot1, color = 'Red', label = 'Efficient frontier')
    plt.plot(risk_plot2, returns_plot2, color = 'Black')
    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns") 
    plt.title("Minimum Variance Curve & Efficient Frontier")
    plt.plot(risk_min_var, mu_min_var, color = 'Blue', marker = 'o',label='Minimum Variance Portfolio')
    plt.annotate('(' + str(round(risk_min_var, 2)) + ', ' + str(round(mu_min_var, 2)) + ')', xy=(risk_min_var, mu_min_var), xytext=(risk_min_var, -0.6))
    plt.legend()
    plt.grid(True)
    plt.show()

    print("PART B-")
    print("Market Portfolio Weights= ", market_portfolio_weights)
    print("Return = ", mu_market)
    print("Risk = ", risk_market * 100, " %")
    print("")

    print("PART C-")
    returns_cml = []
    risk_cml = np.linspace(0, 0.6, num = 5000)
    for i in risk_cml:
        returns_cml.append(mu_rf + (mu_market - mu_rf) * i / risk_market)


    slope, intercept = (mu_market - mu_rf) / risk_market, mu_rf
    print("Equation of CML is:")
    print("y = {:.2f} x + {:.2f}\n".format(slope, intercept))

    plt.plot(risk, returns, color = 'Black', label = 'Minimum Variance Curve')
    plt.plot(risk_cml, returns_cml, color = 'Green', label = 'CML')
    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns") 
    plt.plot(risk_market, mu_market, color = 'Red', marker = 'o',label='Market Portfolio')
    plt.annotate('(' + str(round(risk_market, 2)) + ', ' + str(round(mu_market, 2)) + ')', xy=(risk_market, mu_market), xytext=(0.2, 0.6))
    plt.title("Capital Market Line with Markowitz Efficient Frontier")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(risk_cml, returns_cml)
    plt.xlabel("Risk (sigma)")
    plt.ylabel("Returns") 
    plt.title("Capital Market Line")
    plt.grid(True)
    plt.show()


    print("PART D-")
    stocks_data = ['TCS', 'RIL', 'WIPRO', 'L&T', 'IOC', 'ICICI BANK','ONGC', 'TATA MOTORS', 'HDFC BANK', 'INFOSYS']
    # beta_v = []
    # u_v = []
    # for stock in stocks_data:
    #     beta_v.append((M[stock]-mu_rf)/(mu_market-mu_rf))
    #     u_v.append(M[stock])
    # plt.plot(beta_v,u_v)
    # plt.show()

    beta_k = np.linspace(-1, 1, 5000)
    mu_k = mu_rf + (mu_market - mu_rf) * beta_k
    plt.plot(beta_k, mu_k)
    
    print("Eqn of Security Market Line is:")
    print("mu = {:.2f} beta + {:.2f}".format(mu_market - mu_rf, mu_rf))

    plt.title('Security Market Line for all the 10 assets')
    plt.xlabel("Beta")
    plt.ylabel("Mean Return")
    plt.grid(True)
    plt.show()


if __name__=="__main__":
  main()
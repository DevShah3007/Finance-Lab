import numpy as np
import matplotlib.pyplot as plt
import math
from tabulate import tabulate
from scipy import interpolate

def get_eqn(x, y):
  slope, intercept = [], []
  for i in range(len(x) - 1):
    x1, x2 = x[i], x[i + 1]
    y1, y2 = y[i], y[i + 1]
    slope.append((y2 - y1)/(x2 - x1))
    intercept.append(y1 - slope[-1]*x1)

  return sum(slope)/len(slope), sum(intercept)/len(intercept)

def plot_(M,C,u,clr,lbl):
    w_min = (u@np.linalg.inv(C))/(u@np.linalg.inv(C)@np.transpose(u))
    sigma_min = math.sqrt(w_min@C@np.transpose(w_min))
    u_min = M@np.transpose(w_min)

    weights = []
    no_short_x = []
    no_short_y = []

    markovitz_x = []
    markovitz_y = np.linspace(u_min,0.25,10000)
    nonmarkovitz_x = []
    nonmarkovitz_y = np.linspace(0.05,u_min,10000)

    for uv in markovitz_y:
        m1 = [[1,u@np.linalg.inv(C)@np.transpose(M)],[uv,M@np.linalg.inv(C)@np.transpose(M)]]
        m2 = [[u@np.linalg.inv(C)@np.transpose(u),1],[M@np.linalg.inv(C)@np.transpose(u),uv]]
        m3 = [[u@np.linalg.inv(C)@np.transpose(u),u@np.linalg.inv(C)@np.transpose(M)],[M@np.linalg.inv(C)@np.transpose(u),M@np.linalg.inv(C)@np.transpose(M)]]
        w = (np.linalg.det(m1)*(u@np.linalg.inv(C)) + np.linalg.det(m2)*(M@np.linalg.inv(C)))/np.linalg.det(m3)
        weights.append(w)
        sigma = math.sqrt(w@C@np.transpose(w))
        markovitz_x.append(sigma)
        short = False
        for wt in w:
            if wt<0:
                short = True
        if not short:
            no_short_x.append(sigma)
            no_short_y.append(uv)

    plt.plot(markovitz_x,markovitz_y,color=clr,label=lbl,lw=2)

    for uv in nonmarkovitz_y:
        m1 = [[1,u@np.linalg.inv(C)@np.transpose(M)],[uv,M@np.linalg.inv(C)@np.transpose(M)]]
        m2 = [[u@np.linalg.inv(C)@np.transpose(u),1],[M@np.linalg.inv(C)@np.transpose(u),uv]]
        m3 = [[u@np.linalg.inv(C)@np.transpose(u),u@np.linalg.inv(C)@np.transpose(M)],[M@np.linalg.inv(C)@np.transpose(u),M@np.linalg.inv(C)@np.transpose(M)]]
        w = (np.linalg.det(m1)*(u@np.linalg.inv(C)) + np.linalg.det(m2)*(M@np.linalg.inv(C)))/np.linalg.det(m3)
        sigma = math.sqrt(w@C@np.transpose(w))
        weights.append(w)
        nonmarkovitz_x.append(sigma)
        
    plt.plot(nonmarkovitz_x,nonmarkovitz_y,color=clr,lw=2)

    if len(u) == 3:
        plt.plot(no_short_x,no_short_y,color='Black',label='Minimum Variance Line without short-selling',lw=2)

    plt.xlabel('Risk')
    plt.ylabel('Return')
    plt.grid(True)
    plt.legend()
    return weights

M = [0.1,0.2,0.15]
C = [[0.005,-0.010,0.004],[-0.010, 0.040, -0.002],[0.004, -0.002, 0.023]]
u = [1,1,1]
for w1 in np.linspace(0,1,100):
    for w2 in np.linspace(0,1-w1,100):
        w3 = 1-w1-w2
        w_ = [w1,w2,w3]
        u_x = math.sqrt(np.matmul(np.matmul(w_,C),np.transpose(w_)))
        u_y = np.matmul(M,np.transpose(w_))
        plt.plot(u_x,u_y,marker='.',color='Grey',ms=1)

M = [0.1,0.2]
C = [[0.005,-0.010],[-0.010,0.040]]
u = [1,1]
ww1 = plot_(M,C,u,'Red','Securities 1 and 2')

M = [0.1,0.15]
C = [[0.005,0.004],[0.004,0.023]]
u = [1,1]
ww2= plot_(M,C,u,'Green','Securities 1 and 3')

M = [0.2,0.15]
C = [[0.040,-0.002],[-0.002,0.023]]
u = [1,1]
ww3 = plot_(M,C,u,'Blue','Securities 2 and 3')

M = [0.1,0.2,0.15]
C = [[0.005,-0.010,0.004],[-0.010, 0.040, -0.002],[0.004, -0.002, 0.023]]
u = [1,1,1]
ww4=plot_(M,C,u,'Magenta','Securities 1,2 and 3')

plt.title('Minimum Variance Curves and No-Short Sales Region')
plt.show()

ww4_w1 = np.array([ww[0] for ww in ww4])
ww4_w2 = np.array([ww[1] for ww in ww4])
ww4_w3 = np.array([ww[2] for ww in ww4])

plt.axis([-0.5,1.5,-0.5,1.5])
plt.plot(ww4_w1,ww4_w2,color='Black',label='w1 vs w2',lw=2)
plt.plot(ww4_w1,1-ww4_w1,color='Green',label='w1 + w2 = 1',lw=0.5)
plt.hlines(y=0,xmin=-0.5,xmax=1.5, color='Blue', linestyle='-',label='w1 = 0',lw=0.5)
plt.vlines(x=0,ymin=-0.5,ymax=1.5, color='Red', linestyle='-',label='w2 = 0',lw=0.5)
plt.legend()
plt.grid(True)
plt.title('Weights corresponding to Minimum Variance Curve (w1 vs w2)')
plt.xlabel('w1')
plt.ylabel('w2')
plt.show()

plt.axis([-0.5,1.5,-0.5,1.5])
plt.plot(ww4_w1,ww4_w3,color='Black',label='w1 vs w3',lw=2)
plt.plot(ww4_w1,1-ww4_w1,color='Green',label='w1 + w3 = 1',lw=0.5)
plt.hlines(y=0,xmin=-0.5,xmax=1.5, color='Blue', linestyle='-',label='w1 = 0',lw=0.5)
plt.vlines(x=0,ymin=-0.5,ymax=1.5, color='Red', linestyle='-',label='w3 = 0',lw=0.5)
plt.legend()
plt.grid(True)
plt.title('Weights corresponding to Minimum Variance Curve (w1 vs w3)')
plt.xlabel('w1')
plt.ylabel('w3')
plt.show()

plt.axis([-0.5,1.5,-0.5,1.5])
plt.plot(ww4_w2,ww4_w3,color='Black',label='w2 vs w3',lw=2)
plt.plot(ww4_w2,1-ww4_w2,color='Green',label='w2 + w3 = 1',lw=0.5)
plt.hlines(y=0,xmin=-0.5,xmax=1.5, color='Blue', linestyle='-',label='w2 = 0',lw=0.5)
plt.vlines(x=0,ymin=-0.5,ymax=1.5, color='Red', linestyle='-',label='w3 = 0',lw=0.5)
plt.legend()
plt.grid(True)
plt.title('Weights corresponding to Minimum Variance Curve (w2 vs w3)')
plt.xlabel('w2')
plt.ylabel('w3')
plt.show()

m,c = get_eqn(ww4_w1, ww4_w2)
print("Eqn of line w1 vs w2 is:")
print("w2 = {:.2f} w1 + {:.2f}".format(m, c))
print("")

m,c = get_eqn(ww4_w1, ww4_w3)
print("Eqn of line w1 vs w3 is:")
print("w3 = {:.2f} w1 + {:.2f}".format(m, c))
print("")

m,c = get_eqn(ww4_w2, ww4_w3)
print("Eqn of line w2 vs w3 is:")
print("w3 = {:.2f} w2 + {:.2f}".format(m, c))

import math
import time

S0 = 100
T = 1
r = 0.08
sigma = 0.2
M = 50
K = 100
dt = T/M
u = math.exp(sigma*math.sqrt(dt)+(r-0.5*sigma*sigma)*dt)
d = math.exp(-sigma*math.sqrt(dt)+(r-0.5*sigma*sigma)*dt)
R = math.exp(r*dt)
p = (R-d)/(u-d)
q = 1-p

cache = []

def Markov_LB(s,cm,step):
    if ((s,cm) in cache[step]):
        return cache[step][(s,cm)]
    if step == M:
        cache[step][(s,cm)] = cm - s
    else:
        up = Markov_LB(s*u,max(cm,s*u),step+1)
        down = Markov_LB(s*d,cm,step+1)
        cache[step][(s,cm)] = (p*up + q*down)/R
    return cache[step][(s,cm)]

def Markov_Euro(s,step):
    if s in cache[step]:
        return cache[step][s]
    val = 0
    if step == M:
        val = max(s-K,0)
    else:
        up = Markov_Euro(s*u,step+1)
        down = Markov_Euro(s*d,step+1)
        val = (p*up+q*down)/R
    cache[step][s] = val
    return val

for m in [5,10,25,50,100,500]:
    M = m
    dt = T/M
    u = math.exp(sigma*math.sqrt(dt)+(r-0.5*sigma*sigma)*dt)
    d = math.exp(-sigma*math.sqrt(dt)+(r-0.5*sigma*sigma)*dt)
    R = math.exp(r*dt)
    p = (R-d)/(u-d)
    q = 1-p
    cache.clear() 
    for i in range(M+1):
        cache.append(dict())
    start = time.time()
    ans = Markov_Euro(S0,0)
    end = time.time()
    print(end-start)
    print(ans)

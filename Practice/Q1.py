import math

S0 = 100
T = 1
K = 95
M = 5
r = 0.04
sigma = 0.25
dt = T/M
u = math.exp(sigma*math.sqrt(dt))
d = math.exp(-sigma*math.sqrt(dt))
R = math.exp(r*dt)
p = (R-d)/(u-d)
q = 1-p

cache = []
for i in range(M+1):
    cache.append(dict())

def calculate(smin,current_s,step):
    if (current_s,smin) in cache[step]:
        return cache[step][(current_s,smin)]
    if step == M:
        cache[step][(current_s,smin)] = max(0,K-smin)
    else:
        up = calculate(min(smin,current_s*u),u*current_s,step+1)
        down = calculate(min(smin,current_s*d),d*current_s,step+1)
        cache[step][(current_s,smin)] = (p*up + q*down)/R
    return cache[step][(current_s,smin)]

for m in range(1,11):
    M = m
    dt = T/M
    u = math.exp(sigma*math.sqrt(dt))
    d = math.exp(-sigma*math.sqrt(dt))
    R = math.exp(r*dt)
    p = (R-d)/(u-d)
    q = 1-p
    cache.clear()
    cache = []
    for i in range(M+1):
        cache.append(dict())
    print(calculate(S0,S0,0))
    # if M == 5:
    #     for i in range(M+1):
    #         print("\nLevel ", i)
    #         for val in cache[i]:
    #             print(cache[i][val])
    #         print()
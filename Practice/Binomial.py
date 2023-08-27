import math

S0 = 100
K = 105
T = 5
r = 0.05
sigma = 0.3

def binomial(M):
    dt = T/M
    u = math.exp(sigma*math.sqrt(dt)+(r-0.5*sigma*sigma)*dt)
    d = math.exp(-sigma*math.sqrt(dt)+(r-0.5*sigma*sigma)*dt)
    R = math.exp(r*dt)
    p = (R-d)/(u-d)
    q = 1-p
    call = []
    put = []
    S = S0*(u**M)
    call.append(max(S-K,0))
    put.append(max(K-S,0))
    for i in range(M):
        S/=u
        S*=d
        call.append(max(S-K,0))
        put.append(max(K-S,0))
    for i in range(M):
        for j in range(M-i):
            call[j] = (p*call[j] + q*call[j+1])/R
            put[j] = (p*put[j] + q*put[j+1])/R
    return call[0],put[0]

for m in [1,5,10,20,50]:
    print(binomial(m))


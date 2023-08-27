import math

def lookback_option(S0,T,M,r,sigma):
    dt = (T*1.0)/M
    u = math.e**(sigma*math.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    d = math.e**(-sigma*math.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    R = math.e**(r*dt)
    p = (R-d)/(u-d)

    dp = {}
    m = S0
    for i in range(0,M+1):
        s = S0*(u**M)
        for j in range(0,M+1):
            dp[(s,m)] = m-s
            print(s,m)
            s = (s*d)/u
        m*=u
        print("")

    for k in range(M-1,-1,-1):
        curr = {}
        m = S0
        for i in range(0,k+1):
            s = S0*u**k
            for j in range(0,k+1):
                val = (p*dp.get((s*u,max(m,s*u))) + (1-p)*dp.get((s*d,m)))/R
                curr[(s,m)] = val
                print(s,m,val)
                s = (s*d)/u
            m*=u
            print("")
        dp = curr

    return dp.get((S0,S0))

M = [3]
for m in M:
    option_price = lookback_option(4,1,m,0.08,0.2)
    print("Price of lookback option for M = ", m ," is ", option_price)

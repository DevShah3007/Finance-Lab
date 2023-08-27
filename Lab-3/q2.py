import math

def lookback_option(S0,T,M,r,sigma):
    dt = (T*1.0)/M
    u = math.e**(sigma*math.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    d = math.e**(-sigma*math.sqrt(dt) + (r-0.5*sigma*sigma)*dt)
    R = math.e**(r*dt)
    p = (R-d)/(u-d)
    
    option_price = []
    for i in range(0,M+1):
        temp = []
        for j in range(0,pow(2,i)):
            temp.append(0)
        option_price.append(temp)

    for j in range(pow(2,M)-1,-1,-1):
        temp = bin(j)
        str = temp[2::]
        for k in range(len(str),M):
            str = '0' + str
        curr_price = S0
        curr_max = S0
        for character in str:
            if(character == '1'):
                curr_price*=u
            else:
                curr_price*=d
            curr_max = max(curr_price,curr_max)
        option_price[M][j] = curr_max - curr_price

    for i in range(M-1,-1,-1):
        for j in range(pow(2,i)-1,-1,-1):
            option_price[i][j] = (p*option_price[i+1][2*j+1] + (1-p)*option_price[i+1][2*j])/R
    
    return option_price

M = [5,10,25]
for m in M:
    option_price = lookback_option(100,1,m,0.08,0.2)
    print("Price of lookback option for M = ", m ," is ", option_price[0][0])
    if(m==5):
        print(option_price)
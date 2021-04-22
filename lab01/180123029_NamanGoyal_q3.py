import numpy as np
import matplotlib.pyplot as plt
import math
from math import exp
from tabulate import tabulate

S0 = 100.0
K = 105.0
T = 5
r = 0.05
sigma = 0.3
def generate(s):
    t = T/20
    u = math.exp(sigma*math.sqrt(t)+(r-sigma*sigma/2)*t)
    d = math.exp(-sigma*math.sqrt(t)+(r-sigma*sigma/2)*t)
	
    optionPrices = np.zeros(21)
    q = ((math.exp(r*t)-d)/(u-d))
    for i in range(21):
        optionPrices[i] = S0*(u**(20-i))*(d**(i))
        if s == "call":
            optionPrices[i] = max(0,optionPrices[i]-K)
        else:
            optionPrices[i] = max(0, K-optionPrices[i])

    callprices = np.zeros((21,21))
    putprices = np.zeros((21,21))
    if s == "call":
        callprices[20] = optionPrices
    else:
        putprices[20] = optionPrices
    for i in range(20):
        for j in range(20-i):
            optionPrices[j] = q*optionPrices[j] + (1-q)*optionPrices[j+1]
            optionPrices[j] = optionPrices[j]*math.exp(-r*t)
        if s == "call":
            callprices[i] = optionPrices
        else:
            putprices[i] = optionPrices
    if s == "call":
        return callprices[::-1]
    else:
        return putprices[::-1]
        
time = [0.0, 0.5, 1.0, 1.5, 3.0, 4.5]        
print("Call Prices:")
for i in time:
    callprices = generate("call")
    ll = int(i*4+1)
    print("Time = ", i)
    for j in range(ll):
        print(callprices[ll][ll-j-1])    
    print("")
    
print("Put Prices:")
for i in time:
    putprices = generate("put")
    ll = int(i*4+1)
    print("Time = ", i)    
    for j in range(ll):
        print(putprices[ll][ll-j-1])    
    print("")
    
    

    



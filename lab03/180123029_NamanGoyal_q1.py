import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import time
import math

S0 = 100
T = 1
r = 0.08
sig = 0.2
OptionPrice = []

def optionLookback(M, u, d, p):
    Price = [[[S0, S0]]]
    for i in range(M):
        Q = []
        for j in range(len(Price[i])):
            q = Price[i][j][0]*u*p
            maxq = p*max(Price[i][j][1], q/p)
            Q.append([q, maxq])
            q = Price[i][j][0]*d*(1-p)
            maxq = (1-p)*max(Price[i][j][1], q/(1-p))
            Q.append([q, maxq])
        Price.append(Q)
    ans = 0
    for p1 in Price[len(Price)-1]:
        ans += (p1[1]-p1[0])
    return (ans)*np.exp(-r*T)
stime = time.time()
M = 5
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if(p<0 or p>1):
    print('For M =', M, 'No Arbitrage violated.')
else:
    val = optionLookback(M, u, d, p)
    OptionPrice.append(val)
    print('For M =', M, 'Lookback Option Price is', val)
etime = time.time()
print("The time elapsed for M = 5 => ",etime-stime)
stime = time.time()
M = 10
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if(p<0 or p>1):
    print('For M =', M, 'No Arbitrage violated.')
else:
    val = optionLookback(M, u, d, p)
    OptionPrice.append(val)
    print('For M =', M, 'Lookback Option Price is', val)
etime = time.time()
print("The time elapsed for M = 10 => ",etime-stime)

stime= time.time()
M = 20
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if(p<0 or p>1):
    print('For M =', M, 'No Arbitrage violated.')
else:
    val = optionLookback(M, u, d, p)
    OptionPrice.append(val)
    print('For M =', M, 'Lookback Option Price is', val)
etime= time.time() 
print("The time elapsed for M = 20 => ",etime-stime)

arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
OptionPrice = []

for M in arr:
    t = T/M
    u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
    d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
    p = (np.exp(r*t)-d)/(u-d)
    if(p<0 or p>1):
        print('For M =', M, 'No Arbitrage violated.')
        continue
    val = optionLookback(M, u, d, p)
    OptionPrice.append(val)

plt.plot(arr, OptionPrice)
plt.xlabel('Value of M')
plt.ylabel('Lookback Option Price')
plt.title('Option Price vs M')
plt.show()

def matLookback(M, u, d):
    Price = [[[S0, S0]]]
    for i in range(M):
        Q = []
        for j in range(len(Price[i])):
            q = Price[i][j][0]*u
            maxq = max(Price[i][j][1], q)
            Q.append([q, maxq])
            q = Price[i][j][0]*d
            maxq = max(Price[i][j][1], q)
            Q.append([q, maxq])
        Price.append(Q)
    return Price

M = 5
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if(p<0 or p>1):
    print('For M =', M, 'No Arbitrage violated.')
else:
    print("Option Price value for different time intervals are")
    mat = matLookback(M, u, d)
    price_list = []
    for P in mat[len(mat)-1]:
        price_list.append(P[1]-P[0])
    for i in range(6):
        print('For t =', (5-i)*t)
        print(price_list)
        temp = []
        for i in range(int(len(price_list)/2)):
            temp.append((p*price_list[2*i]+(1-p)*price_list[2*i+1])*np.exp(-r*t))
        price_list = temp


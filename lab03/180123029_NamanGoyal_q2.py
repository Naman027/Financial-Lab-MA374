import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import time

S0 = 100
T = 1
r = 0.08
sig = 0.2
Map = {}

def MarkovBasedefficientAlgo(u, d, p, M, n, S, ma):
    if (S, ma) in Map:
        return Map[(S, ma)]
    if n == M:
        Map[(S, ma)] = ma-S
        return ma-S
    U = MarkovBasedefficientAlgo(u, d, p, M, n+1, S*u, max(ma, S*u))
    D = MarkovBasedefficientAlgo(u, d, p, M, n+1, S*d, max(ma, S*d))
    ans = (p*U + (1-p)*D)*np.exp(-r*T/M)
    Map[(S, ma)] = ans
    return ans
stime = time.time()
M = 5
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if(p<0 or p>1):
    print('For M =', M, 'No Arbitrage Violated.')
else:
    Map.clear()
    ans = MarkovBasedefficientAlgo(u, d, p, M, 0, S0, S0)
    print('For M =', M, 'Lookback option Value is', ans)
etime = time.time()
print("Time elapsed for M = 5 => ",etime-stime)

stime = time.time()
M = 10
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if(p<0 or p>1):
    print('For M =', M, 'No Arbitrage Violated.')
else:
    Map.clear()
    ans = MarkovBasedefficientAlgo(u, d, p, M, 0, S0, S0)
    print('For M =', M, 'Lookback option Value is', ans)
etime = time.time()
print("Time elapsed for M = 10 => ",etime-stime)

stime = time.time()
M = 15
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if(p<0 or p>1):
    print('For M =', M, 'No Arbitrage Violated.')
else:
    Map.clear()
    ans = MarkovBasedefficientAlgo(u, d, p, M, 0, S0, S0)
    print('For M =', M, 'Lookback option Value is', ans)
etime = time.time()
print("Time elapsed for M = 15 => ",etime-stime)

stime = time.time()
M = 25
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if(p<0 or p>1):
    print('For M =', M, 'No Arbitrage Violated.')
else:
    Map.clear()
    ans = MarkovBasedefficientAlgo(u, d, p, M, 0, S0, S0)
    print('For M =', M, 'Lookback option Value is', ans)
etime = time.time()
print("Time elapsed for M = 25 => ",etime-stime)

stime = time.time()
M = 50
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if(p<0 or p>1):
    print('For M =', M, 'No Arbitrage Violated.')
else:
    Map.clear()
    ans = MarkovBasedefficientAlgo(u, d, p, M, 0, S0, S0)
    print('For M =', M, 'Lookback option Value is', ans)
etime = time.time()
print("Time elapsed for M = 50 => ",etime-stime)




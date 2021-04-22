import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
import time

S0 = 100
K = 100
T = 1
r = 0.08
sig = 0.2

def MarkovBasedefficientAlgoEuro(S0, K, M, r, sig, u, d, p):
    eff_arr = [0]*(M+1)
    for i in range(M+1):
        eff_arr[i] = max(S0*(u**i)*(d**(M-i)) - K, 0)
    for i in range(M):
        for j in range(M-i):
            eff_arr[j] = ((1-p)*eff_arr[j] + p*eff_arr[j+1])*np.exp(-r*T/M)
    ans = eff_arr[0]
    return ans

def NormalcalcEuro(S0, K, M, r, sig, u, d, p):
    Price = [[[S0, K]]]
    for i in range(M):
        Q = []
        for el in Price[i]:
            Q.append([el[0]*u*p, el[1]*p])
            Q.append([el[0]*d*(1-p), el[1]*(1-p)])
        Price.append(Q)
    ans = 0
    for el in Price[len(Price)-1]:
        ans += max(el[0]-el[1], 0)
    return ans*np.exp(-r*T/M)

M = 10
stime = time.time()
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if M < 25:
    val = NormalcalcEuro(S0, K, M, r, sig, u, d, p)
    print('Value of European Call for M = ', M, 'Normally', val)
else:
    print("Normal method can't handle value of M =", M)
val = MarkovBasedefficientAlgoEuro(S0, K, M, r, sig, u, d, p)
print('Value of European Call for M = ', M, 'Efficiently', val)
etime = time.time()
print("Time elapsed for M = 10 => ",etime-stime)

M = 15
stime = time.time()
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if M < 25:
    val = NormalcalcEuro(S0, K, M, r, sig, u, d, p)
    print('Value of European Call for M = ', M, 'Normally', val)
else:
    print("Normal method can't handle value of M =", M)
val = MarkovBasedefficientAlgoEuro(S0, K, M, r, sig, u, d, p)
print('Value of European Call for M = ', M, 'Efficiently', val)
etime = time.time()
print("Time elapsed for M = 15 => ",etime-stime)

M = 25
stime = time.time()
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if M < 25:
    val = NormalcalcEuro(S0, K, M, r, sig, u, d, p)
    print('Value of European Call for M = ', M, 'Normally', val)
else:
    print("Normal method can't handle value of M =", M)
val = MarkovBasedefficientAlgoEuro(S0, K, M, r, sig, u, d, p)
print('Value of European Call for M = ', M, 'Efficiently', val)
etime = time.time()
print("Time elapsed for M = 25 => ",etime-stime)

M = 50
stime = time.time()
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if M < 25:
    val = NormalcalcEuro(S0, K, M, r, sig, u, d, p)
    print('Value of European Call for M = ', M, 'Normally', val)
else:
    print("Normal method can't handle value of M =", M)
val = MarkovBasedefficientAlgoEuro(S0, K, M, r, sig, u, d, p)
print('Value of European Call for M = ', M, 'Efficiently', val)
etime = time.time()
print("Time elapsed for M = 50 => ",etime-stime)

M = 100
stime = time.time()
t = T/M
u = np.exp(sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
d = np.exp(-sig*np.sqrt(t) + (r-0.5*sig*sig)*t)
p = (np.exp(r*t)-d)/(u-d)
if M < 25:
    val = NormalcalcEuro(S0, K, M, r, sig, u, d, p)
    print('Value of European Call for M = ', M, 'Normally', val)
else:
    print("Normal method can't handle value of M =", M)
val = MarkovBasedefficientAlgoEuro(S0, K, M, r, sig, u, d, p)
print('Value of European Call for M = ', M, 'Efficiently', val)
etime = time.time()
print("Time elapsed for M = 100 => ",etime-stime)
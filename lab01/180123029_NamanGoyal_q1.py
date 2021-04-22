import numpy as np
import matplotlib.pyplot as plt
from math import exp
from tabulate import tabulate

S0 = 100.0
K = 105.0
T = 5
r = 0.05
sigma = 0.3

M = [1, 5, 10, 20, 50, 100, 200, 400]

optionPrices = []
for step in M:
    t = T/step
    u = exp(sigma*(t**0.5) + (r - 0.5*(sigma**2))*t)
    d = exp(-sigma*(t**0.5) + (r - 0.5*(sigma**2))*t)

    q = (exp(r*t) - d) / (u-d)
    size = step+1

    stockPrice = np.zeros((size,size))
    P = np.zeros((size,size))
    payoff = np.zeros((size,size))

    stockPrice[0][0] = S0
    P[0][0] = 1
    for j in range(1, size):
        for i in range(j):
            stockPrice[i][j] = stockPrice[i][j-1]*u
            stockPrice[i+1][j] = stockPrice[i][j-1]*d
            P[i][j] = P[i][j-1]*u
            P[i+1][j] = P[i][j-1]*d
    
    stockPrice = np.round(stockPrice, decimals = 2)
    
    for i in range(size):
        payoff[i][size-1] = max(0, K - P[i][size-1]*S0)

    for j in range(size-2, -1, -1):
        for i in range(j+1):
            payoff[i][j] = exp(-r*t)*(q*payoff[i][j+1] + (1-q)*payoff[i+1][j+1])
    
    optionPrices.append(payoff[0][0])

print("For the Put Option Pricing")
print(" ");
priceInitial = []
for i in range(len(M)):
    priceInitial.append([M[i],optionPrices[i]]);

print(tabulate(priceInitial,headers = ["Value of M","Initial Option Price"]));

print("------------------------------------------------------------------------------------")

optionPrices = []
for step in M:
    t = T/step
    u = exp(sigma*(t**0.5) + (r - 0.5*(sigma**2))*t)
    d = exp(-sigma*(t**0.5) + (r - 0.5*(sigma**2))*t)

    q = (exp(r*t) - d) / (u-d)
    size = step+1

    stockPrice = np.zeros((size,size))
    P = np.zeros((size,size))
    payoff = np.zeros((size,size))

    stockPrice[0][0] = S0
    P[0][0] = 1
    for j in range(1, size):
        for i in range(j):
            stockPrice[i][j] = stockPrice[i][j-1]*u
            stockPrice[i+1][j] = stockPrice[i][j-1]*d
            P[i][j] = P[i][j-1]*u
            P[i+1][j] = P[i][j-1]*d
    
    stockPrice = np.round(stockPrice, decimals = 2)
    
    for i in range(size):
        payoff[i][size-1] = max(0, P[i][size-1]*S0-K)

    for j in range(size-2, -1, -1):
        for i in range(j+1):
            payoff[i][j] = exp(-r*t)*(q*payoff[i][j+1] + (1-q)*payoff[i+1][j+1])
    
    optionPrices.append(payoff[0][0])

print("For the Call Option Pricing")
print(" ")
priceInitial = []
for i in range(len(M)):
    priceInitial.append([M[i],optionPrices[i]]);

print(tabulate(priceInitial,headers = ["Value of M","Initial Option Price"]));



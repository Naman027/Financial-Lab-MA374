import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt, log, exp
from mpl_toolkits.mplot3d import Axes3D
from pandas.io.pytables import DataCol
from scipy.stats import norm

def StockNext(mu,sigma,S0):
    delta_t = 1/252
    W = np.random.normal(0, 1)
    drift = delta_t * (mu - (sigma**2)/2)
    diffusion = sigma*W*np.sqrt(delta_t)
    Sk = S0 * np.exp(drift + diffusion)
    return Sk

def StockPriceEst(mu, sigma, S0, T):
    prices = [S0]
    curr = S0
    for i in range(T):
        curr = StockNext(mu, sigma, curr)
        prices.append(curr)

    return prices

def payoff(mu,sigma,K,S0,N):
    Sk = S0
    Sum = 0
    for i in range(1,N+1):
        Sk = StockNext(mu,sigma,Sk)
        Sum += Sk
    meanStockPrice = Sum/(N+1)	
    Put_payoff = max(K - meanStockPrice, 0)
    Call_payoff = max(meanStockPrice - K, 0)
    return Call_payoff, Put_payoff

def Asian_Price(mu,sigma,K,S0,r,T):
    CallPayoff = []
    PutPayoff = []
    for i in range(100):
        Call_payoff, Put_payoff = payoff(mu,sigma,K,S0,T)
        CallPayoff.append(Call_payoff)
        PutPayoff.append(Put_payoff)
    
    CallPayoff = np.array(CallPayoff)
    PutPayoff = np.array(PutPayoff)
    CallOptionPrice = CallPayoff * np.exp(-r*T/252)
    PutOptionPrice = PutPayoff * np.exp(-r*T/252)

    return np.mean(CallOptionPrice), np.mean(PutOptionPrice)

def Q1():
    print("-------------------Q1-------------------")
    print("")
    S0 = 100
    mu = 0.1
    sigma = 0.2
    r = 0.05
    T = 126
    dates = range(T+1)

    for i in range(10):
        prices = StockPriceEst(mu, sigma, S0, T)
        plt.plot(dates, prices)

    plt.xlabel('Time (t)')
    plt.ylabel('Stock Price')
    plt.title('Stock Prices using GBM in real world')
    plt.savefig("Q1_RealStock")
    plt.clf()

    for i in range(10):
        prices = StockPriceEst(r, sigma, S0, T)
        plt.plot(dates, prices)

    plt.xlabel('Time (t)')
    plt.ylabel('Stock Price')
    plt.title('Stock prices using GBM in Risk-Neutral world')
    plt.savefig("Q1_RiskNeutralStock")
    plt.clf()

    K = [90, 105, 110]
    for k in K:
        print("For K = %d"%(k))
        call, put = Asian_Price(r, sigma, k, S0, r, T)
        print("Call Price : %.6f"%(call))
        print("Put Price  : %.6f"%(put))
        print("")
    
    K = 105
    # Varying K
    K1 = np.linspace(85, 115, 50)
    call_prices = []
    put_prices = []

    for k1 in K1:   
        call, put = Asian_Price(r, sigma, k1, S0, r, T)
        call_prices.append(call)
        put_prices.append(put)

    plt.plot(K1, call_prices, label = "Call")
    plt.plot(K1, put_prices, label = "Put")
    plt.xlabel("Strike Price")
    plt.ylabel("Option Price")
    plt.title("Varying Strike Price")
    plt.legend(loc = 'best')
    plt.savefig("Q1_VaryK")
    plt.clf()

    # Varying sigma
    Sigma1 = np.linspace(0.05, 0.35, 50)
    call_prices = []
    put_prices = []

    for sigma1 in Sigma1:   
        call, put = Asian_Price(r, sigma1, K, S0, r, T)
        call_prices.append(call)
        put_prices.append(put)

    plt.plot(Sigma1, call_prices, label = "Call")
    plt.plot(Sigma1, put_prices, label = "Put")
    plt.xlabel("Sigma")
    plt.ylabel("Option Price")
    plt.title("Varying Sigma")
    plt.legend(loc = 'best')
    plt.savefig("Q1_VarySigma")
    plt.clf()

    # Varying S0
    S1 = np.linspace(85, 115, 50)
    call_prices = []
    put_prices = []

    for s1 in S1:   
        call, put = Asian_Price(r, sigma, K, s1, r, T)
        call_prices.append(call)
        put_prices.append(put)

    plt.plot(S1, call_prices, label = "Call")
    plt.plot(S1, put_prices, label = "Put")
    plt.xlabel("S0")
    plt.ylabel("Option Price")
    plt.title("Varying S0")
    plt.legend(loc = 'best')
    plt.savefig("Q1_VaryS0")
    plt.clf()
    print("-------------------------------------------------------")
    print("")

def StockNext_Antithetic(mu,sigma,S0):
    delta_t = 1/252
    W1 = np.random.normal(0, 1)
    drift = delta_t * (mu - (sigma**2)/2)
    diffusion = sigma*W1*np.sqrt(delta_t)
    Sk1 = S0 * np.exp(drift + diffusion)
    W2 = -np.random.normal(0, 1)
    drift = delta_t * (mu - (sigma**2)/2)
    diffusion = sigma*W2*np.sqrt(delta_t)
    Sk2 = S0 * np.exp(drift + diffusion)
    return (Sk1 + Sk2)/2

def payoff_varreduction(mu, sigma, K, S0, N, reducedVar = True):
    Sk = S0
    Sum = 0
    for i in range(1,N+1):
        if reducedVar:
            Sk = StockNext_Antithetic(mu,sigma,Sk)
        else:
            Sk = StockNext(mu, sigma, Sk)
        Sum += Sk
    meanStockPrice = Sum/(N+1)	
    Put_payoff = max(K - meanStockPrice, 0)
    Call_payoff = max(meanStockPrice - K, 0)
    return Call_payoff, Put_payoff

def Asian_Price_VarReduction(mu,sigma,K,S0,r,T, reducedVar = True):
    CallPayoff = []
    PutPayoff = []
    for i in range(100):
        Call_payoff, Put_payoff = payoff_varreduction(mu,sigma,K,S0,T, reducedVar)
        CallPayoff.append(Call_payoff)
        PutPayoff.append(Put_payoff)
    CallPayoff = np.array(CallPayoff)
    PutPayoff = np.array(PutPayoff)
    CallOptionPrice = CallPayoff * np.exp(-r*T/252)
    PutOptionPrice = PutPayoff * np.exp(-r*T/252)
    return CallOptionPrice, PutOptionPrice

def Q2():
    print("-------------------Q2-------------------")
    print("")
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 126
    K = [90, 105, 110]

    for k in K:
        print("For K = %d"%(k))
        call, put = Asian_Price_VarReduction(r, sigma, k, S0, r, T, reducedVar=False)
        call_red, put_red = Asian_Price_VarReduction(r, sigma, k, S0, r, T, reducedVar=True)
        print("Without Variance Reduction:: Call Price : %.6f and Variance = %.6f"%(np.mean(call), np.var(call)))
        print("With Variance Reduction:: Call Price : %.6f and Variance = %.6f"%(np.mean(call_red), np.var(call_red)))
        print("Without Variance Reduction:: Put Price : %.6f and Variance = %.6f"%(np.mean(put), np.var(put)))
        print("With Variance Reduction:: Put Price : %.6f and Variance = %.6f"%(np.mean(put_red), np.var(put_red)))
        print("")

    K = 105
    # Varying K
    K1 = np.linspace(85, 115, 50)
    call_prices = []
    put_prices = []

    for k1 in K1:   
        call, put = Asian_Price_VarReduction(r, sigma, k1, S0, r, T)
        call_prices.append(np.mean(call))
        put_prices.append(np.mean(put))

    plt.plot(K1, call_prices, label = "Call")
    plt.plot(K1, put_prices, label = "Put")
    plt.xlabel("Strike Price")
    plt.ylabel("Option Price")
    plt.title("Varying Strike Price")
    plt.legend(loc = 'best')
    plt.savefig("Q2_VaryK")
    plt.clf()

    # Varying sigma
    Sigma1 = np.linspace(0.05, 0.35, 50)
    call_prices = []
    put_prices = []

    for sigma1 in Sigma1:   
        call, put = Asian_Price_VarReduction(r, sigma1, K, S0, r, T)
        call_prices.append(np.mean(call))
        put_prices.append(np.mean(put))

    plt.plot(Sigma1, call_prices, label = "Call")
    plt.plot(Sigma1, put_prices, label = "Put")
    plt.xlabel("Sigma")
    plt.ylabel("Option Price")
    plt.title("Varying Sigma")
    plt.legend(loc = 'best')
    plt.savefig("Q2_VarySigma")
    plt.clf()

    # Varying S0
    S1 = np.linspace(85, 115, 50)
    call_prices = []
    put_prices = []

    for s1 in S1:   
        call, put = Asian_Price_VarReduction(r, sigma, K, s1, r, T)
        call_prices.append(np.mean(call))
        put_prices.append(np.mean(put))

    plt.plot(S1, call_prices, label = "Call")
    plt.plot(S1, put_prices, label = "Put")
    plt.xlabel("S0")
    plt.ylabel("Option Price")
    plt.title("Varying S0")
    plt.legend(loc = 'best')
    plt.savefig("Q2_VaryS0")
    plt.clf()
    print("-------------------------------------------------------")
    print("")

Q1()
Q2()


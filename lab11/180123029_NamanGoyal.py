import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt, log, exp
from numpy import exp, log, sqrt, linspace
from mpl_toolkits.mplot3d import Axes3D
from pandas.io.pytables import DataCol
from scipy.stats import norm

def Vasicek(beta, mu, sig, r0, T, t):
    b = beta*mu
    a = beta
    B = (1/a)*(1 - exp(-a*(T-t)))
    A = (B-T+t)*(a*b - 0.5*sig*sig)/(a*a) - (sig*sig*B*B)/(4*a)
    p = exp(A - B*r0)
    y = -log(p)/(T-t)
    return y

def Q1():
    print("-------------------Q1-------------------")
    print("")
    Beta = [5.9, 3.9, 0.1]
    Mu = [0.2, 0.1, 0.4]
    Sig = [0.3, 0.3, 0.11]
    R0 = [0.1, 0.2, 0.1]
    T = linspace(1, 10, 10)
    print("The Term Structure upto 10 time units")
    for i in range(3):
        beta = Beta[i]
        mu = Mu[i]
        sig = Sig[i]
        r0 = R0[i]
        Y = [Vasicek(beta, mu, sig, r0, T1,  0) for T1 in T]
        plt.plot(T, Y)
        plt.xlabel("Time")
        plt.ylabel("Yield")
        plt.title("Vasicek Term structure for {beta = %0.1f, mu = %0.1f, sig = %0.2f, r0 = %0.1f}"%(beta, mu, sig, r0))
        plt.savefig("Q1_10_time-units_%d"%i)
        plt.clf()

    T = linspace(1/252, 500/252, 500)
    R0s = linspace(0.1, 1, 10)
    print("The Yield v/s Maturity for upto 500 time units")
    for i in range(3):
        beta = Beta[i]
        mu = Mu[i]
        sig = Sig[i]
        for r0 in R0s:
            Y = [Vasicek(beta, mu, sig, r0, T1,  0) for T1 in T]
            plt.plot(T, Y, label = 'r = %.1f'%r0)
        plt.xlabel("Time")
        plt.ylabel("Yield")
        plt.title("Vasicek Yield Curves vs Maturity for {beta = %0.1f, mu = %0.1f, sig = %0.2f}"%(beta, mu, sig))
        plt.legend(loc = 'best')
        plt.savefig("Q1_500_time-units_%d"%i)
        plt.clf()
    print("-------------------------------------------------------")
    print("")

def CIR_Calcs(beta, mu, sig, r0, T, t):
    a = beta
    b = mu 
    h = sqrt(a**2 + 2*(sig**2))
    A = ((2*h*exp((a+h)*(T-t)/2))/(2*h + (a+h)*(exp((T-t)*h)-1)))**((2*a*b)/(sig**2))
    B = (2*(exp((T-t)*h)-1))/(2*h + (a+h)*(exp((T-t)*h)-1))
    p = A*exp(-r0*B)
    y = -log(p)/(T-t)
    return y

def Q2():
    print("-------------------Q2-------------------")
    print("")
    Beta = [0.02, 0.7, 0.06]
    Mu = [0.7, 0.1, 0.09]
    Sig = [0.02, 0.3, 0.5]
    R0 = [0.1, 0.2, 0.02]
    T = linspace(1, 10, 10)
    print("The Term Structure upto 10 time units")
    for i in range(3):
        beta = Beta[i]
        mu = Mu[i]
        sig = Sig[i]
        r0 = R0[i]
        Y = [CIR_Calcs(beta, mu, sig, r0, T1,  0) for T1 in T]
        plt.plot(T, Y)
        plt.xlabel("Time")
        plt.ylabel("Yield")
        plt.title("CIR Term structure for {beta = %0.2f, mu = %0.2f, sig = %0.2f, r0 = %0.1f}"%(beta, mu, sig, r0))
        plt.savefig("Q2_10_time-units_%d"%i)
        plt.clf()

    T = linspace(1, 600, 600)
    R0s = linspace(0.1, 1, 10)
    beta = 0.02
    mu = 0.7
    sig = 0.02
    print("The Yield v/s Maturity for upto 600 time units")
    for r0 in R0s:
        Y = [CIR_Calcs(beta, mu, sig, r0, T1,  0) for T1 in T]
        plt.plot(T, Y, label = 'r = %.1f'%r0)
    plt.xlabel("Time")
    plt.ylabel("Yield")
    plt.title("CIR Yield Curves vs Maturity for {beta = %0.2f, mu = %0.1f, sig = %0.2f}"%(beta, mu, sig))
    plt.legend(loc = 'best')
    plt.savefig("Q2_600_time-units")
    plt.clf()
    print("-------------------------------------------------------")
    print("")

Q1()
Q2()


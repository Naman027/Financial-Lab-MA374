import time
import csv
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time
import pandas as pd
from pandas import read_csv

M=[0.1,0.2,0.15]
C=[ [0.005,-0.010,0.004],
    [-0.010,0.040,-0.002],
    [0.004,-0.002,0.023]]

u=[1,1,1]
invc=np.linalg.inv(C)

def Sigma_W_calc(Mu):
    W=[]
    Sigma=[]
    uicm=np.dot(u,invc)
    uicm=np.dot(uicm,M)
    micm=np.dot(M,invc)
    micm=np.dot(micm,M)
    uicu=np.dot(u,invc)
    uicu=np.dot(uicu,u)
    micu=np.dot(M,invc)
    micu=np.dot(micu,u)
    for mu in Mu:
        d1=np.linalg.det([[1,uicm],[mu,micm]])
        d2=np.linalg.det([[uicu,1],[micu,mu]])
        num=d1*np.dot(u,invc)+d2*np.dot(M,invc)
        d3=np.linalg.det([[uicu,uicm],[micu,micm]])
        num=num/d3
        temp=np.dot(num,C)
        temp=np.dot(temp,num)
        W.append(num)
        Sigma.append(mt.sqrt(temp))
    return Sigma,W

def marketPortfolio(rfr):
    ru=[]
    w=[]
    for i in range(0,len(u)):
        ru.append(rfr*u[i])
        w.append(M[i]-ru[i])
    w = np.dot(w,invc)
    w = w/np.sum(w)
    return w

Mean=np.linspace(0.005,0.5,6000)
Sigma,W=Sigma_W_calc(Mean)
def PartA():
    print("Solution (a): See Graph")
    plt.plot(Sigma,Mean,'r');
    plt.title("Efficient frontier");
    plt.xlabel('Volatility');
    plt.ylabel('Return');
    plt.axvline(x=0.15, color="green")
    plt.axhline(y=0.18, color="green")
# Calc the intersection
    plt.text(0.15,-0.01,'x = 0.15')
    plt.text(-0.02,0.18,'y = 0.18')
    plt.show()

def PartB():
    print("Solution (b)")
    print("Value of portfolio for 10 different values of efficient frontier")
    for i in range(0,5000,500):
        print("Return: ",round(Mean[i],6)," Risk: ",round(Sigma[i],6)," W1: ",round(W[i][0],6)," W2: ",round(W[i][1],6)," W3: ",round(W[i][2],6))

def PartC():
    print("Solution (c)")
    print("At 15 percent Risk, Max and Min Portfolios:")
    val = 0.0001
    for i in range(len(Sigma)):
        if abs(Sigma[i]-0.15) < val:
            print("Return: ",round(Mean[i],6)," Risk: ",round(Sigma[i],6)," W1: ",round(W[i][0],6)," W2: ",round(W[i][1],6)," W3: ",round(W[i][2],6))

M_Weights=marketPortfolio(0.1)
M_Mean=np.dot(M_Weights,M)
M_Sigma=np.dot(np.dot(M_Weights,C),M_Weights)
M_Sigma=mt.sqrt(M_Sigma)
Sigma18,W18=Sigma_W_calc([0.18])

def PartD():
    print("Solution (d)")
    print("Portfolio (Without Riskfree Assets) at 18 percent")
    print("Return: ",round(0.18,6)," Risk: ",round(Sigma18[0],6)," W1: ",round(W18[0][0],6)," W2: ",round(W18[0][1],6)," W3: ",round(W18[0][2],6))
    plt.plot(M_Sigma,M_Mean,'-o')

def PartE():
    print("Solution (e)")
    print("Risk Free Return 10% Market Portfolio")
    print("Return: ",round(M_Sigma,6)," Risk: ",round(M_Mean,6)," W1: ",M_Weights[0]," W2: ",round(M_Weights[1],6)," W3: ",round(M_Weights[2],6))
    plt.scatter(M_Sigma, M_Mean, color="b",label="Market portfolio")
    plt.scatter(0,0.1,color="black",label="zero risk portfolio")
    plt.plot(Sigma,Mean,'r',label="Efficient Frontier");
    ymax = 0.5
    xmax = M_Sigma+M_Sigma*(ymax-M_Mean)/(M_Mean-0.1)
    X=[0,xmax]
    Y=[0.1,ymax]
    plt.plot(X,Y,"g",label="Capital Market Line")
    plt.title("Efficient Frontier and CAPM");
    plt.xlabel('Volatility');
    plt.ylabel('Return');
    plt.legend()
    plt.show()

def PartF():
    print("Solution (f)")
    risk1 = 0.1
    c1 = risk1/M_Sigma
    w1 = np.append(c1*M_Weights, (1-c1)*1)
    print("Portfolio(with risky and riskfree assets) at 0.1 percent risk :")
    print("Risk Free asset Weightage:",round(w1[3],6))
    print("Risky asset Weightge:",round(w1[0],6),round(w1[1],6),round(w1[2],6),)

    risk2 = 0.25
    c2 = risk2/M_Sigma
    w2 = np.append(c2*M_Weights, (1-c2)*1)
    print("Portfolio (Including Risky and Riskfree Assets) at 0.25 percent risk :")
    print("Risk Free asset Weightage:",round(w2[3],6))
    print("Risky asset Weightage:",round(w2[0],6),round(w2[2],6),round(w2[2],6))

PartA()
print("--------------------------------------------------------------------")
print("")
PartB()
print("--------------------------------------------------------------------")
print("")
PartC()
print("--------------------------------------------------------------------")
print("")
PartD()
print("--------------------------------------------------------------------")
print("")
PartE()
print("--------------------------------------------------------------------")
print("")
PartF()
print("--------------------------------------------------------------------")
print("")



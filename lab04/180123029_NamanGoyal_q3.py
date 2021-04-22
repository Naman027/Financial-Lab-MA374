import csv
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time
import pandas as pd
from pandas import read_csv

Companies_Stocks=['AAPL.csv','AMZN.csv','FB.csv','GOOG(Alphabet).csv','IBM.csv','NKE.csv','SBINNS.csv','TATAMOTORS.csv','TSLA.csv','SAM.csv']
Companies = ['APPLE','AMAZON','FACEBOOK','ALPHABET','IBM','NIKE','SBI','TATA MOTORS','TESLA','SAMSUNG']
dim = len(Companies_Stocks)
length = 60
stock=[]
Mean = []
rate=0.05
M=[0.1,0.2,0.15]
C=	[[0.005,-0.010,0.004],
    [-0.010,0.040,-0.002],
    [0.004,-0.002,0.023]]

fields = ['Close', 'Open']
for name in Companies_Stocks:
	dataframe = read_csv(name, 
				skipinitialspace = True, squeeze = True, 
				usecols = fields)
	dataframe = (dataframe['Close'] - dataframe['Open'])
	data=[]
	for i in range(0,60):
		data.append(dataframe[i])
	data = np.array(data)
	stock.append(data)
	Mean.append(np.mean(data))

stock = np.array(stock)
M = np.array(Mean)
C = np.cov(stock)

stock_df = pd.DataFrame(np.transpose(np.round(stock, 3)))
stock_df.to_csv('Data.csv', index=False, header=Companies_Stocks)

u=[1,1,1,1,1,1,1,1,1,1]
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

def marketPortfolio(rate):
    ru=[]
    w=[]
    for i in range(0,len(u)):
        ru.append(rate*u[i])
        w.append(M[i]-ru[i])
    w = np.dot(w,invc)
    w = w/np.sum(w)
    return w

Mean=np.linspace(-4,6,5000)
Sigma,W=Sigma_W_calc(Mean)

def PartA():
	print("Solution (a)")
	plt.plot(Sigma,Mean,'r');
	plt.title("Efficient frontier");
	plt.xlabel('Volatility');
	plt.ylabel('Return');
	plt.show()

M_Weights=marketPortfolio(0.05)
M_Mean=np.dot(M_Weights,M)
M_Sigma=np.dot(np.dot(M_Weights,C),M_Weights)
M_Sigma=mt.sqrt(M_Sigma)

def PartB():
	print("Solution (b)")
	plt.plot(M_Sigma,M_Mean,'-o')
	print("Market Portfolio for Risk Free Return 5%")
	print("Return: ",round(M_Sigma,6)," Risk: ",round(M_Mean,6)," W1: ",M_Weights[0]," W2: ",round(M_Weights[1],6)," W3: ",round(M_Weights[2],6)," W4: ",round(M_Weights[3],6)," W5: ",round(M_Weights[4],6)," W6: ",round(M_Weights[5],6)," W7: ",round(M_Weights[6],6)," W8: ",round(M_Weights[7],6)," W9: ",round(M_Weights[8],6)," W10: ",round(M_Weights[9],6))


def PartC():
	print("Solution (c)")
	print("The plot for The Capital Line is shown")
	plt.scatter(M_Sigma, M_Mean, color="b",label="maket portfolio")
	plt.scatter(0,0.1,color="black",label="zero risk portfolio")
	plt.plot(Sigma,Mean,'r',label="Markowitz Efficient Frontier");
	ymin = -6
	ymax = 6
	xmin = M_Sigma+M_Sigma*(ymin-M_Mean)/(M_Mean-0.05)
	X=[xmin,0]
	Y=[ymin,0.05]
	plt.plot(X,Y,"g",label="Capital Market Line")
	plt.title("Efficient Frontier and CAPM Line");
	plt.xlabel('Volatility');
	plt.ylabel('Return');
	plt.legend()
	plt.show()

def PartD():
	print("Solution (d)")
	print("The plots for the Security Line for 10 different assets are shown")
	for i in range(len(M)):
		X = np.linspace(-3, 3, 100)
		Y = rate + (M[i] - rate)*X
		plt.plot(X, Y, label=Companies[i])
	plt.title("Security Market Lines")
	plt.xlabel("Beta")
	plt.ylabel("Mean Return")
	plt.legend()
	plt.show()

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

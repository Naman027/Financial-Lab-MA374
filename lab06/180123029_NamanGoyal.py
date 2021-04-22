import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

def readCSV(file):
    df = pd.read_csv(file)
    df.set_index('Date',inplace=True)
    st_data = df.to_dict()
    for key, vals in st_data.items():
        st_data[key] = list(vals.values())
    return st_data

bse_daily_data = readCSV('bse_daily.csv')
bse_week_data = readCSV('bse_weekly.csv')
bse_month_data = readCSV('bse_monthly.csv')
nse_daily_data = readCSV('nse_daily.csv')
nse_week_data = readCSV('nse_weekly.csv')
nse_month_data = readCSV('nse_monthly.csv')

def Q1():
    print("-------------------Q1-------------------")
    print("")
    print("FOR BSE INDEXED COMPANIES -------------------")
    for comp, value in bse_daily_data.items():
        X = range(len(value))
        Y = value
        plt.plot(X,Y,label = comp)
    plt.xlabel("Day No")
    plt.ylabel('Stock Price')
    plt.title("BSE Daily Data")
    plt.legend(loc = 'upper left')
    plt.savefig("BSE Daily Data")
    plt.clf()
    for comp, value in bse_week_data.items():
        X = range(len(value))
        Y = value
        plt.plot(X,Y,label = comp)
    plt.xlabel("Week No")
    plt.ylabel('Stock Price')
    plt.title("BSE Weekly Data")
    plt.legend(loc = 'upper left')
    plt.savefig("BSE Weekly Data")
    plt.clf()
    for comp, value in bse_month_data.items():
        X = range(len(value))
        Y = value
        plt.plot(X,Y,label = comp)
    plt.xlabel("Month No")
    plt.ylabel('Stock Price')
    plt.title("BSE Monthly Data")
    plt.legend(loc = 'upper left')
    plt.savefig("BSE Monthly Data")
    plt.clf()
    print("FOR NSE INDEXED COMPANIES -------------------")
    for comp, value in nse_daily_data.items():
        X = range(len(value))
        Y = value
        plt.plot(X,Y,label = comp)
    plt.xlabel("Day No")
    plt.ylabel('Stock Price')
    plt.title("NSE Daily Data")
    plt.legend(loc = 'upper left')
    plt.savefig("NSE Daily Data")
    plt.clf()
    for comp, value in nse_week_data.items():
        X = range(len(value))
        Y = value
        plt.plot(X,Y,label = comp)
    plt.xlabel("Week No")
    plt.ylabel('Stock Price')
    plt.title("NSE Weekly Data")
    plt.legend(loc = 'upper left')
    plt.savefig("NSE Weekly Data")
    plt.clf()
    for comp, value in nse_month_data.items():
        X = range(len(value))
        Y = value
        plt.plot(X,Y,label = comp)
    plt.xlabel("Month No")
    plt.ylabel('Stock Price')
    plt.title("NSE Monthly Data")
    plt.legend(loc = 'upper left')
    plt.savefig("NSE Monthly Data")
    plt.clf()
    print("-------------------------------------------------------")

def Returns(St):
    Ri = []
    n = len(St)
    for i in range(1,n):
        ri = (St[i] - St[i-1])/St[i-1]
        Ri.append(ri)
    mean = np.mean(Ri)
    sigma = np.sqrt(np.var(Ri))
    Ri_cap = [((ri-mean)/sigma) for ri in Ri]
    return Ri_cap

def N(x,mu,var):
    return (1/(np.sqrt(2*var*np.pi)))*(np.exp((-0.5)*(x-mu)*(x-mu)/var))

def Q2():
    print("-------------------Q2-------------------")
    print("")
    print("NSE and BSE Daily Data")
    for comp, value in bse_daily_data.items():
        returns = Returns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 140, edgecolor = 'black', density = 1, label = 'Returns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('BSE Daily Returns for %s'%(comp))
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('BSE_Daily_' + comp)
        plt.clf()
    
    for comp, value in nse_daily_data.items():
        returns = Returns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 140, edgecolor = 'black', density = 1, label = 'Returns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('NSE Daily Returns for %s'%(comp))
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('NSE_Daily_' + comp)
        plt.clf()

    print("NSE and BSE Weekly Data")
    for comp, value in bse_week_data.items():
        returns = Returns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 35, edgecolor = 'black', density = 1, label = 'Returns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('BSE Weekly Returns for %s'%(comp))
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('BSE_Weekly_' + comp)
        plt.clf()
    
    for comp, value in nse_week_data.items():
        returns = Returns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 35, edgecolor = 'black', density = 1, label = 'Returns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('NSE Weekly Returns for %s'%(comp))
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('NSE_Weekly_' + comp)
        plt.clf()

    print("NSE and BSE Monthly Data")
    for comp, value in bse_month_data.items():
        returns = Returns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 15, edgecolor = 'black', density = 1, label = 'Returns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('BSE Monthly Returns for %s'%(comp))
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('BSE_Monthly_' + comp)
        plt.clf()

    for comp, value in nse_month_data.items():
        returns = Returns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 15, edgecolor = 'black', density = 1, label = 'Returns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('NSE Monthly Returns for %s'%(comp))
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('NSE_Monthly_' + comp)
        plt.clf()
    print("-------------------------------------------------------")

def LogReturns(St):
    Ri = []
    n = len(St)
    for i in range(1,n):
        ri = (St[i] - St[i-1])/St[i-1]
        Ri.append(ri)
    Ri = [np.log(1 + r) for r in Ri]
    mu = np.mean(Ri)
    sig = np.sqrt(np.var(Ri))
    Ri_cap = [((ri-mu)/sig) for ri in Ri]
    return Ri_cap

def Q3():
    print("-------------------Q3-------------------")
    print("")
    print("NSE and BSE Daily Data")
    for comp, value in bse_daily_data.items():
        returns = LogReturns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 140, edgecolor = 'black', density = 1, label = 'LogReturns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('BSE Daily LogReturns for %s'%(comp))
        plt.xlabel('LogReturns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('BSE_Dailylog_' + comp)
        plt.clf()
    
    for comp, value in nse_daily_data.items():
        returns = LogReturns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 140, edgecolor = 'black', density = 1, label = 'LogReturns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('NSE Daily LogReturns for %s'%(comp))
        plt.xlabel('LogReturns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('NSE_Dailylog_' + comp)
        plt.clf()

    print("NSE and BSE Weekly Data")
    for comp, value in bse_week_data.items():
        returns = LogReturns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 35, edgecolor = 'black', density = 1, label = 'LogReturns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('BSE Weekly LogReturns for %s'%(comp))
        plt.xlabel('LogReturns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('BSE_Weeklylog_' + comp)
        plt.clf()
    
    for comp, value in nse_week_data.items():
        returns = LogReturns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 35, edgecolor = 'black', density = 1, label = 'LogReturns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('NSE Weekly LogReturns for %s'%(comp))
        plt.xlabel('LogReturns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('NSE_Weeklylog_' + comp)
        plt.clf()

    print("NSE and BSE Monthly Data")
    for comp, value in bse_month_data.items():
        returns = LogReturns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 15, edgecolor = 'black', density = 1, label = 'LogReturns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('BSE Monthly LogReturns for %s'%(comp))
        plt.xlabel('LogReturns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('BSE_Monthlylog_' + comp)
        plt.clf()

    for comp, value in nse_month_data.items():
        returns = LogReturns(value)
        X = np.linspace(-4,4,10000)
        Y = N(X,0,1)
        fig = plt.figure(figsize=(5.6,4.2))
        plt.hist(returns, bins = 15, edgecolor = 'black', density = 1, label = 'LogReturns')
        plt.plot(X,Y,color = 'blue', label = 'N(0,1) Curve')
        plt.title('NSE Monthly LogReturns for %s'%(comp))
        plt.xlabel('LogReturns')
        plt.ylabel('Frequency')
        plt.legend(loc = 'best')
        plt.savefig('NSE_Monthlylog_' + comp)
        plt.clf()
    print("-------------------------------------------------------")

def nextPredictValue(mu,sigma,St):
    Xt = np.log(St)
    Z = np.random.normal(0,1)
    N = np.random.poisson(0.2)
    # From Monte Carlo Simulations Lab Sem V
    M = 0
    if N != 0:
        for i in range(N):
            Y = np.random.lognormal(mu,sigma)
            M += np.log(Y)
    Xt_new = Xt + (mu - (sigma**2)/2)*1 + sigma*Z*np.sqrt(1) + M
    St_new = np.exp(Xt_new)
    return St_new

def MuSigma(St):
    Ri = []
    n = len(St)
    for i in range(1,n):
        ri = (St[i] - St[i-1])/St[i-1]
        Ri.append(ri)
    Ri = [np.log(1 + r) for r in Ri]
    mu = np.mean(Ri)
    sig = np.sqrt(np.var(Ri))
    return mu,sig

def predictPrice(data, days_req):
    cur = data[-1]
    mu, sig = MuSigma(data)
    prices = []
    for i in range(days_req):
        nextP = nextPredictValue(mu,sig,cur)
        prices.append(nextP)
        cur = nextP
    return prices

def Q4Q5():
    print("-------------------Q4 && Q5-------------------")
    print("")

    bse_daily_data_till2017 = readCSV('bse_daily_till2017.csv')
    bse_week_data_till2017 = readCSV('bse_weekly_till2017.csv')
    bse_month_data_till2017 = readCSV('bse_monthly_till2017.csv')
    nse_daily_data_till2017 = readCSV('nse_daily_till2017.csv')
    nse_week_data_till2017 = readCSV('nse_weekly_till2017.csv')
    nse_month_data_till2017 = readCSV('nse_monthly_till2017.csv')
    
    print("NSE and BSE Daily Data")
    for comp, value in bse_daily_data_till2017.items():
        data_req = len(bse_daily_data[comp])-len(bse_daily_data_till2017[comp])
        predicted_price = predictPrice(value, data_req)
        value.extend(predicted_price)        
        X = range(len(value))
        fig = plt.figure(figsize=(5.6,4.2))
        plt.plot(X, value, label = 'Predicted Price')
        plt.plot(X, bse_daily_data[comp], label = 'Actual Price')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("BSE Daily Price for %s"%(comp))
        plt.legend(loc = 'best')
        plt.savefig("Predicted_Daily_BSE_%s"%(comp))
        plt.clf()

    for comp, value in nse_daily_data_till2017.items():
        data_req = len(nse_daily_data[comp])-len(nse_daily_data_till2017[comp])
        predicted_price = predictPrice(value, data_req)
        value.extend(predicted_price)        
        X = range(len(value))

        fig = plt.figure(figsize=(5.6,4.2))
        plt.plot(X, value, label = 'Predicted Price')
        plt.plot(X, nse_daily_data[comp], label = 'Actual Price')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("NSE Daily Price for %s"%(comp))
        plt.legend(loc = 'best')
        plt.savefig("Predicted_Daily_NSE_%s"%(comp))
        plt.clf()

    print("NSE and BSE Weekly Data")
    for comp, value in bse_week_data_till2017.items():
        data_req = len(bse_week_data[comp])-len(bse_week_data_till2017[comp])
        predicted_price = predictPrice(value, data_req)
        value.extend(predicted_price)        
        X = range(len(value))

        fig = plt.figure(figsize=(5.6,4.2))
        plt.plot(X, value, label = 'Predicted Price')
        plt.plot(X, bse_week_data[comp], label = 'Actual Price')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("BSE Weekly Price for %s"%(comp))
        plt.legend(loc = 'best')
        plt.savefig("Predicted_Weekly_BSE_%s"%(comp))
        plt.clf()

    for comp, value in nse_week_data_till2017.items():
        data_req = len(nse_week_data[comp])-len(nse_week_data_till2017[comp])
        predicted_price = predictPrice(value, data_req)
        value.extend(predicted_price)        
        X = range(len(value))

        fig = plt.figure(figsize=(5.6,4.2))
        plt.plot(X, value, label = 'Predicted Price')
        plt.plot(X, nse_week_data[comp], label = 'Actual Price')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("NSE Weekly Price for %s"%(comp))
        plt.legend(loc = 'best')
        plt.savefig("Predicted_Weekly_NSE_%s"%(comp))
        plt.clf()

    print("NSE and BSE Monthly Data")
    for comp, value in bse_month_data_till2017.items():
        data_req = len(bse_month_data[comp])-len(bse_month_data_till2017[comp])
        predicted_price = predictPrice(value, data_req)
        value.extend(predicted_price)        
        X = range(len(value))

        fig = plt.figure(figsize=(5.6,4.2))
        plt.plot(X, value, label = 'Predicted Price')
        plt.plot(X, bse_month_data[comp], label = 'Actual Price')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("BSE Monthly Price for %s"%(comp))
        plt.legend(loc = 'best')
        plt.savefig("Predicted_Monthly_BSE_%s"%(comp))
        plt.clf()

    for comp, value in nse_month_data_till2017.items():
        data_req = len(nse_month_data[comp])-len(nse_month_data_till2017[comp])
        predicted_price = predictPrice(value, data_req)
        value.extend(predicted_price)        
        X = range(len(value))

        fig = plt.figure(figsize=(5.6,4.2))
        plt.plot(X, value, label = 'Predicted Price')
        plt.plot(X, nse_month_data[comp], label = 'Actual Price')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("NSE Monthly Price for %s"%(comp))
        plt.legend(loc = 'best')
        plt.savefig("Predicted_Monthly_NSE_%s"%(comp))
        plt.clf()
    print("-------------------------------------------------------")

Q1()
Q2()
Q3()
Q4Q5()

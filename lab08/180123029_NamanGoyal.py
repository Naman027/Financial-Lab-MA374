import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'figure.max_open_warning': 0})

def readCSV(file):
    df = pd.read_csv(file)
    df.set_index('Date',inplace=True)
    st_data = df.to_dict()
    for key, vals in st_data.items():
        st_data[key] = list(vals.values())
    return st_data

def fun_Ctx(S0, t, K, r, sig, T=0.5):
    t = T-t
    if sig*t!=0:
    	d1 = (np.log(S0/K)+t*(r+(sig*sig)/2))/(sig*(t**0.5))
    else: 
    	d1 = 0
    d2 = d1-sig*(t**0.5)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    C = S0*Nd1 - K*np.exp(-r*t)*Nd2
    return C

def fun_Ptx(S0, t, K, r, sig, T=0.5):
    t = T-t
    if sig*t!=0:
    	d1 = (np.log(S0/K)+t*(r+(sig*sig)/2))/(sig*(t**0.5))
    else:
    	d1 = 0
    d2 = d1-sig*(t**0.5)
    Nd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(-d2)
    P = K*np.exp(-r*t)*Nd2 - S0*Nd1
    return P

def Historic_Volatility(prices, duration):
    Prices = prices[-duration:]
    R = []
    for i in range(1, len(Prices)):
        ri = (Prices[i] - Prices[i-1])/Prices[i-1]
        R.append(ri)
    var = np.var(R)
    sig_d = np.sqrt(var)
    sig_a = np.sqrt(252)*sig_d
    return sig_a

bse_data = readCSV('bse_daily.csv')
nse_data = readCSV('nse_daily.csv')

def Q1():
    print("-------------------Q1-------------------")
    print("")

    workingDays = 20
    print("For BSE Companies")
    for company, prices in bse_data.items():
        vol = Historic_Volatility(prices, workingDays)
        print("Historical Volatility for %s : %.5f"%(company, vol))
    print("")
    print("---------------")
    print("")
    print("For NSE Companies")
    for company, prices in nse_data.items():
        vol = Historic_Volatility(prices, workingDays)
        print("Historical Volatility for %s : %.5f"%(company, vol))

    print("-------------------------------------------------------")

def Q2():
    print("-------------------Q2-------------------")
    print("")
    r = 0.05
    T = 0.5
    t = 0
    workingDays = 20
    A = np.arange(0.5, 1.51, 0.1)
    print("For BSE Companies (K = S0)")
    for company, prices in bse_data.items():
        S0 = prices[-1]
        K = S0
        hist_Vol = Historic_Volatility(prices, workingDays)
        call_price = fun_Ctx(S0,t,K,r,hist_Vol,T)
        put_price = fun_Ptx(S0,t,K,r,hist_Vol,T)
        print(company)
        print("Call Price = %.5f"%call_price)
        print("Put Price =  %.5f"%put_price)
        print("")

        data = {"Strike":[],"Call Price":[],"Put Price":[]}
        for el in A:
            K = el*S0
            call_price = fun_Ctx(S0,t,K,r,hist_Vol,T)
            put_price = fun_Ptx(S0,t,K,r,hist_Vol,T)
            data["Strike"].append(K)
            data["Call Price"].append(call_price)
            data["Put Price"].append(put_price)

        df = pd.DataFrame(data)
        df.to_csv(f"BSE_{company}.csv", header=True, index=False)
    
    print("")
    print("---------------")
    print("")
    print("For NSE Companies (K = S0)")
    for company, prices in nse_data.items():
        S0 = prices[-1]
        K = S0
        hist_Vol = Historic_Volatility(prices, workingDays)
        call_price = fun_Ctx(S0,t,K,r,hist_Vol,T)
        put_price = fun_Ptx(S0,t,K,r,hist_Vol,T)
        print(company)
        print("Call Price = %.5f"%call_price)
        print("Put Price =  %.5f"%put_price)
        print("")

        data = {"Strike":[],"Call Price":[],"Put Price":[]}
        for el in A:
            K = el*S0
            call_price = fun_Ctx(S0,t,K,r,hist_Vol,T)
            put_price = fun_Ptx(S0,t,K,r,hist_Vol,T)
            data["Strike"].append(K)
            data["Call Price"].append(call_price)
            data["Put Price"].append(put_price)

        df = pd.DataFrame(data)
        df.to_csv(f"NSE_{company}.csv", header=True, index=False)
    print("-------------------------------------------------------")

def Q3():
    print("-------------------Q3-------------------")
    print("")

    r = 0.05
    T = 0.5
    t = 0
    workingDays = 20
    A = np.arange(0.5, 1.51, 0.1)
    print("For BSE Companies")
    for company, prices in bse_data.items():
        months = range(1, 61)
        B = []
        for i in range(1, 61):
            n_days = i*workingDays
            b = Historic_Volatility(prices, n_days)
            B.append(b)
        plt.plot(months, B, label = company)
    plt.xlabel("Month")
    plt.ylabel("Historical Volatility")
    plt.title("BSE Historical Volatility vs Month")
    plt.legend(loc = "best")
    plt.savefig("BSE_HistoricVolatility")
    plt.clf()

    for company, prices in bse_data.items():
        S0 = prices[-1]
        months = range(1,61)
        fig = plt.figure(figsize=(5.6,4.2))
        for el in A:
            K = el*S0
            call_prices= []
            for i in range(1,61):
                dayscount = i*workingDays
                hist_Vol = Historic_Volatility(prices, dayscount)
                call = fun_Ctx(S0,t,K,r,hist_Vol,T)
                call_prices.append(call)
            plt.plot(months, call_prices, label = "K = %.1f*S0"%el)

        plt.xlabel("Month")
        plt.ylabel("Call Price")
        plt.title(f"BSE Call Price vs Historical Volatility for {company}")
        plt.legend(loc = "best")
        plt.savefig(f"BSE_{company}_CallPrice")
        plt.clf()

    for company, prices in bse_data.items():
        S0 = prices[-1]
        months = range(1,61)
        fig = plt.figure(figsize=(5.6,4.2))
        for el in A:
            K = el*S0
            put_prices= []
            for i in range(1,61):
                dayscount = i*workingDays
                hist_Vol = Historic_Volatility(prices, dayscount)
                put = fun_Ptx(S0,t,K,r,hist_Vol,T)
                put_prices.append(put)
            plt.plot(months, put_prices, label = "K = %.1f*S0"%el)
        plt.xlabel("Month")
        plt.ylabel("Put Price")
        plt.title(f"BSE Put Price vs Historical Volatility for {company}")
        plt.legend(loc = "best")
        plt.savefig(f"BSE_{company}_PutPrice")
        plt.clf()

    print("")
    print("---------------")
    print("")
    print("For NSE Companies")
    for company, prices in nse_data.items():
        months = range(1, 61)
        B = []
        for i in range(1, 61):
            dayscount = i*workingDays
            b = Historic_Volatility(prices, dayscount)
            B.append(b)
        plt.plot(months, B, label = company)
    plt.xlabel("Month")
    plt.ylabel("Historical Volatility")
    plt.title("NSE Historical Volatility vs Month")
    plt.legend(loc = "best")
    plt.savefig("NSE_HistoricVolatility")
    plt.clf()

    for company, prices in nse_data.items():
        S0 = prices[-1]
        months = range(1,61)
        fig = plt.figure(figsize=(5.6,4.2))
        for el in A:
            K = el*S0
            call_prices= []
            for i in range(1,61):
                dayscount = i*workingDays
                hist_Vol = Historic_Volatility(prices, dayscount)
                call = fun_Ctx(S0,t,K,r,hist_Vol,T)
                call_prices.append(call)
            plt.plot(months, call_prices, label = "K = %.1f*S0"%el)

        plt.xlabel("Month")
        plt.ylabel("Call Price")
        plt.title(f"NSE Call Price vs Historical Volatility for {company}")
        plt.legend(loc = "best")
        plt.savefig(f"NSE_{company}_CallPrice")
        plt.clf()

    for company, prices in nse_data.items():
        S0 = prices[-1]
        months = range(1,61)
        fig = plt.figure(figsize=(5.6,4.2))
        for el in A:
            K = el*S0
            put_prices= []
            for i in range(1,61):
                dayscount = i*workingDays
                hist_Vol = Historic_Volatility(prices, dayscount)
                put = fun_Ptx(S0,t,K,r,hist_Vol,T)
                put_prices.append(put)
            plt.plot(months, put_prices, label = "K = %.1f*S0"%el)
        plt.xlabel("Month")
        plt.ylabel("Put Price")
        plt.title(f"NSE Put Price vs Historical Volatility for {company}")
        plt.legend(loc = "best")
        plt.savefig(f"NSE_{company}_PutPrice")
        plt.clf()
    print("-------------------------------------------------------")

Q1()
Q2()
Q3()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import erf, sqrt, log, exp
from mpl_toolkits.mplot3d import Axes3D
from pandas.io.pytables import DataCol
from scipy.stats import norm

def readCSV(filename):
    df = pd.read_csv(filename)
    call_data = []
    put_data = []
    for index, row in df.iterrows():
        call_price = row['Call Price']
        put_price = row['Put Price']
        strike_price = row['Strike Price']
        maturity = (60 - (index%61))%61
        call_data.append({'Price': call_price, 'Strike': strike_price, 'Maturity':maturity})
        put_data.append({'Price': put_price, 'Strike': strike_price, 'Maturity':maturity})
    return call_data, put_data

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

def fun_Ctx(t, S0, T, sig, K, r):
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

def fun_Ptx(t, S0, T, sig, K, r):
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

def impliedVolatility(call_option_price,put_option_price, s, K, r, T):
    eps = 1e-6
    t = 0 
    # Call Option 
    upper_vol = 100.0
    max_vol = 100.0
    lower_vol = 0.0000001
    call_price = np.inf
    while abs(call_price - call_option_price) >= eps:
        mid_vol = (upper_vol + lower_vol)/2
        call_price = fun_Ctx(0, s, T, mid_vol, K, r)
        lower_price = fun_Ctx(0, s, T, lower_vol, K, r)
        if (lower_price - call_option_price) * (call_price - call_option_price) > 0:
            lower_vol = mid_vol
        else:
            upper_vol = mid_vol
        if mid_vol > max_vol - 0.01:
            mid_vol = 1e-6
            break
    call_vol = mid_vol

    # Put Price
    upper_vol = 100.0
    lower_vol = 0.000001
    min_vol = 1e-5
    put_price = np.inf
    while abs(put_price - put_option_price) >= eps:
        mid_vol = (upper_vol + lower_vol)/2
        put_price = fun_Ptx(0, s, T, mid_vol, K, r)
        upper_price = fun_Ptx(0, s, T, upper_vol, K, r)
        if (upper_price - put_option_price) * (put_price - put_option_price) > 0:
            upper_vol = mid_vol
        else:
            lower_vol = mid_vol
        if mid_vol > max_vol - 0.01 or mid_vol < min_vol:
            mid_vol = 1e-6
            break

    put_vol = mid_vol
    return call_vol, put_vol

def plot2D(data, optType, fixedMaturity, fixedStrike, company):
    # Maturity vs Option Price
    for strike in fixedStrike:
        maturity = []
        prices = []
        for row in data:
            if row['Strike'] == strike:
                mat = row['Maturity']
                price = row['Price']
                maturity.append(mat)
                prices.append(price)
        plt.plot(maturity, prices, label = 'Strike Price = %d'%strike)
    plt.xlabel("Maturity")
    plt.ylabel("%s Option Price")
    plt.title("%s Option Price vs Maturity for %s"%(optType, company))
    plt.legend(loc = 'best')
    plt.savefig(f"Q2_maturity_{company}_{optType}")
    plt.clf()

    # Strike Price vs Option Price 
    for fixedMat in fixedMaturity:
        strikePrices = []
        prices = []
        for row in data:
            if row['Maturity'] == fixedMat:
                strike = row['Strike']
                price = row['Price']
                strikePrices.append(strike)
                prices.append(price)

        plt.plot(strikePrices, prices, label = 'Maturity = %d'%fixedMat)
    plt.xlabel("Strike Price")
    plt.ylabel("%s Option Price")
    plt.title("%s Option Price vs Strike Price for %s"%(optType, company))
    plt.legend(loc = 'best')
    plt.savefig(f"Q2_strike_{company}_{optType}")
    plt.clf()

def plot3D(data,optType, company):
    maturity = []
    prices = []
    strikePrices = []
    for row in data:
        mat = row['Maturity']
        price = row['Price']
        strike = row['Strike']
        maturity.append(mat)
        prices.append(price)
        strikePrices.append(strike)
    ax = plt.axes(projection='3d')
    ax.scatter(np.array(maturity), np.array(strikePrices), np.array(prices))
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Strike Price")
    ax.set_zlabel("%s Option Price"%optType) 
    ax.set_title("%s Option Price for %s"%(optType, company))
    plt.savefig(f"Q2_3D_{company}_{optType}")
    plt.clf()

files = ['NIFTYoptiondata.csv', 'stockoptiondata_CIPLA.csv',  'stockoptiondata_ICICI.csv' , 'stockoptiondata_ITC.csv']
companies = ['NIFTY', 'CIPLA',  'ICICI', 'ITC']

def Q2():
    print("-------------------Q2-------------------")
    print("")
    fixedMaturity = [1, 30, 60]
    for filename,company in zip(files,companies):
        call_data, put_data  = readCSV(filename)
        n = len(call_data)
        fixedStrike = [call_data[0]['Strike'], call_data[-1]['Strike'], call_data[n//2]['Strike']]
        plot3D(call_data, "Call", company)
        plot3D(put_data, "Put", company)
        plot2D(call_data, "Call", fixedMaturity, fixedStrike, company)
        plot2D(put_data, "Put", fixedMaturity, fixedStrike, company)
    print("-------------------------------------------------------")

def Q3():
    print("-------------------Q3-------------------")
    print("")
    df = pd.read_csv('nsedata1.csv')
    r = 0.05
    for filename, company in zip(files, companies):
        call_data, put_data = readCSV(filename)
        fixed_strike = call_data[0]['Strike']
        fixed_maturity = 30
        call_vols = []
        put_vols = []
        mats = []
        strikes = []
        call_vols2D = []
        put_vols2D = []
        mats2D = []
        call_vols2D1 = []
        put_vols2D1 = []
        strikes2D1 = []
        for row_call, row_put in zip(call_data, put_data):
            mat = row_call['Maturity']
            if 0 < mat < 60:
                T = mat/252
                K = row_call['Strike']
                call_option_price = row_call['Price']
                put_option_price = row_put['Price']
                s = float(df[company][59-mat])
                call_vol, put_vol = impliedVolatility(call_option_price, put_option_price, s, K, r, T)
                call_vols.append(call_vol)
                put_vols.append(put_vol)
                mats.append(mat)
                strikes.append(K)
                if K == fixed_strike:
                    call_vols2D.append(call_vol)
                    put_vols2D.append(put_vol)
                    mats2D.append(mat)
                if mat == fixed_maturity:
                    call_vols2D1.append(call_vol)
                    put_vols2D1.append(put_vol)
                    strikes2D1.append(K)
        plt.scatter(mats2D, call_vols2D, label = 'Call', s = 1)
        plt.scatter(mats2D, put_vols2D, label = 'Put', s = 1)
        plt.legend(loc = 'best')
        plt.title(company)
        plt.xlabel('Maturity')
        plt.ylabel('Implied Volatility')
        plt.title(f"Implied Volatility vs Maturity Strike Price = {fixed_strike} for {company}")
        plt.savefig(f"Q3_volatility_2D_maturity_{company}")
        plt.clf()
        plt.plot(strikes2D1, call_vols2D1, label = 'Call')
        plt.plot(strikes2D1, put_vols2D1, label = 'Put')
        plt.legend(loc = 'best')
        plt.title(company)
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title(f"Implied Volatility vs Strike Price Maturity = {fixed_maturity} for {company} ")
        plt.savefig(f"Q3_volatility_2D_strike_{company}")
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.scatter(np.array(mats), np.array(strikes), np.array(call_vols))
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Strike Price")
        ax.set_zlabel("Implied Volatility") 
        ax.set_title(f"Call Implied Volatility for {company}")
        plt.savefig(f"Q3_volatility_call_3D_{company}")
        plt.clf()
        ax = plt.axes(projection='3d')
        ax.scatter(np.array(mats), np.array(strikes), np.array(put_vols))
        ax.set_xlabel("Maturity")
        ax.set_ylabel("Strike Price")
        ax.set_zlabel("Implied Volatility") 
        ax.set_title(f"Put Implied Volatility for {company}")
        plt.savefig(f"Q3_volatility_put_3D_{company}")
        plt.clf()
    print("-------------------------------------------------------")

def Q4():
    print("-------------------Q4-------------------")
    print("")
    df = pd.read_csv('nsedata1.csv')
    for company in companies:
        prices = list(df[company])
        vols = []
        T = range(2, 59)
        for t in T:
            vol = Historic_Volatility(prices, t)
            vols.append(vol)
        plt.plot(T, vols)
        plt.xlabel("Maturity")
        plt.ylabel("Historical Volatility")
        plt.title("Historical Volatility vs Maturity for %s"%company)
        plt.savefig(f"Q4_historical_volatility_for_{company}")
        plt.clf()
    print("-------------------------------------------------------")

Q2()
Q3()
Q4()



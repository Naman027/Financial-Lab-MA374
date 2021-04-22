import numpy as np
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd

def fun_Ctx(S0, t, K, r, sig, T=1):
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

def fun_Ptx(S0, t, K, r, sig, T=1):
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

def Q3():
    print("---------Q3----------")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    t = np.linspace(0, 0.8, 50)
    s = np.linspace(0.8, 1.2, 50)
    t, s = np.meshgrid(t, s)
    C = np.zeros_like(t)
    P = np.zeros_like(t)
    for i in range(50):
        for j in range(50):
            C[i][j] = fun_Ctx(s[i][j], t[i][j], 1, 0.05, 0.6)
            P[i][j] = fun_Ptx(s[i][j], t[i][j], 1, 0.05, 0.6)

    # Plotting a 3D Graph
    surfC = ax.plot_surface(t, s, C, color='b', label='Call Option')
    surfC._facecolors2d=surfC._facecolor3d
    surfC._edgecolors2d=surfC._edgecolor3d
    surfP = ax.plot_surface(t, s, P, color='g', label='Put Option')
    surfP._facecolors2d=surfP._facecolor3d
    surfP._edgecolors2d=surfP._edgecolor3d
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Initial Stock Price')
    ax.set_zlabel('Option Price')
    plt.title('C(t,x) && P(t,x) v/s x && t')
    plt.show()
    
Q3()

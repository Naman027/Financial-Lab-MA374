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

T = 0.5
r = 0.05
sig = 0.6
K = 1

def Q4_varyK():
    print("Varying Strike Price");
    print("Part I <-> 2D Plot")
    S = np.linspace(0.8, 1.2, 5)
    K = np.linspace(0.8, 1.2, 50)
    
    for s in S:
        C = [fun_Ctx(s, T, k, r, sig) for k in K]
        plt.plot(K, C, label='S = {}'.format(s))
    plt.xlabel('K')
    plt.ylabel('Call Option Price')
    plt.title('C(k) v/s K')
    plt.legend()
    plt.show()
    plt.clf()

    for s in S:
        P = [fun_Ptx(s, T, k, r, sig) for k in K]
        plt.plot(K, P, label='S = {}'.format(s))
    plt.xlabel('K')
    plt.ylabel('Put Option Price')
    plt.title('P(k) v/s K')
    plt.legend()
    plt.show()

    print("Part II <-> Table")
    print("")
    S = 0.8
    K = np.linspace(0.8, 1.2, 5)
    C = [fun_Ctx(S, T, k, r, sig) for k in K]
    P = [fun_Ptx(S, T, k, r, sig) for k in K]
    table = []
    for i in range(len(K)):
        table.append([K[i], C[i], P[i]])
    df = pd.DataFrame(table, columns = ['K','Call Price','Put Price'])
    print(df)
    print("")

    print("Part III <-> 3D Plot")
    K = np.linspace(0.8, 1.2, 25)
    S = np.linspace(0.8, 1.2, 25)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    K, S = np.meshgrid(K, S)
    C = np.zeros_like(K)
    P = np.zeros_like(K)

    for i in range(25):
        for j in range(25):
            C[i][j] = fun_Ctx(S[i][j], T, K[i][j], r, sig)
            P[i][j] = fun_Ptx(S[i][j], T, K[i][j], r, sig)

    surfC = ax.plot_surface(K, S, C, color='b', label='Call Option')
    surfC._facecolors2d=surfC._facecolor3d
    surfC._edgecolors2d=surfC._edgecolor3d
    surfP = ax.plot_surface(K, S, P, color='g', label='Put Option')
    surfP._facecolors2d=surfP._facecolor3d
    surfP._edgecolors2d=surfP._edgecolor3d
    ax.legend()
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Initial Stock Price')
    ax.set_zlabel('Option Price')
    plt.title('C(K,x) && P(K,x) v/s x && K')
    plt.show()
    print("")

def Q4_varyR():
    print("Varying Rate");
    print("Part I <-> 2D Plot")
    S = np.linspace(0.8, 1.2, 5)
    R = np.linspace(0.01, 0.1, 50)
    
    for s in S:
        C = [fun_Ctx(s, T, K, r, sig) for r in R]
        plt.plot(R, C, label='S = {}'.format(s))
    plt.xlabel('Rate')
    plt.ylabel('Call Option Price')
    plt.title('C(r) v/s rate')
    plt.legend()
    plt.show()
    plt.clf()

    for s in S:
        P = [fun_Ptx(s, T, K, r, sig) for r in R]
        plt.plot(R, P, label='S = {}'.format(s))
    plt.xlabel('Rate')
    plt.ylabel('Put Option Price')
    plt.title('P(r) v/s Rate')
    plt.legend()
    plt.show()

    print("Part II <-> Table")
    print("")
    S = 0.8
    R = np.linspace(0.01, 0.1, 5)
    C = [fun_Ctx(S, T, K, r, sig) for r in R]
    P = [fun_Ptx(S, T, K, r, sig) for r in R]
    table = []
    for i in range(len(R)):
        table.append([R[i], C[i], P[i]])
    df = pd.DataFrame(table, columns = ['Rate','Call Price','Put Price'])
    print(df)
    print("")

    print("Part III <-> 3D Plot")
    r = np.linspace(0.01, 0.1, 25)
    S = np.linspace(0.8, 1.2, 25)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    r, S = np.meshgrid(r, S)
    C = np.zeros_like(r)
    P = np.zeros_like(r)
    for i in range(25):
        for j in range(25):
            C[i][j] = fun_Ctx(S[i][j], T, K, r[i][j], sig)
            P[i][j] = fun_Ptx(S[i][j], T, K, r[i][j], sig)
    surfC = ax.plot_surface(r, S, C, color='b', label='Call Option')
    surfC._facecolors2d=surfC._facecolor3d
    surfC._edgecolors2d=surfC._edgecolor3d
    surfP = ax.plot_surface(r, S, P, color='g', label='Put Option')
    surfP._facecolors2d=surfP._facecolor3d
    surfP._edgecolors2d=surfP._edgecolor3d
    ax.legend()
    ax.set_xlabel('Rate')
    ax.set_ylabel('Initial Stock Price')
    ax.set_zlabel('Option Price')
    plt.title('C(r,x) && P(r,x) v/s x && r')
    plt.show()
    print("")

def Q4_varySigma():
    print("Varying Sigma");
    print("Part I <-> 2D Plot")
    S = np.linspace(0.8, 1.2, 5)
    Sig = np.linspace(0.1, 1.0, 50)
    for s in S:
        C = [fun_Ctx(s, T, K, r, sig) for sig in Sig]
        plt.plot(Sig, C, label='S = {}'.format(s))
    plt.xlabel('Standard Deviation (Sigma) ')
    plt.ylabel('Call Option Price')
    plt.title('C(sig) v/s Sigma')
    plt.legend()
    plt.show()
    plt.clf()

    for s in S:
        P = [fun_Ptx(s, T, K, r, sig) for sig in Sig]
        plt.plot(Sig, P, label='S = {}'.format(s))
    plt.xlabel('Standard Deviation (Sigma)')
    plt.ylabel('Put Option Price')
    plt.title('P(sig) v/s Sigma')
    plt.legend()
    plt.show()

    print("Part II <-> Table")
    print("")
    S = 0.8
    Sig = np.linspace(0.1, 1.0, 5)
    C = [fun_Ctx(S, T, K, r, sig) for sig in Sig]
    P = [fun_Ptx(S, T, K, r, sig) for sig in Sig]
    table = []
    for i in range(len(Sig)):
        table.append([Sig[i], C[i], P[i]])
    df = pd.DataFrame(table, columns = ['Sigma','Call Price','Put Price'])
    print(df)
    print("")

    print("Part III <-> 3D Plot")
    Sig = np.linspace(0.1, 1.0, 25)
    S = np.linspace(0.8, 1.2, 25)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Sig, S = np.meshgrid(Sig, S)
    C = np.zeros_like(Sig)
    P = np.zeros_like(Sig)
    for i in range(25):
        for j in range(25):
            C[i][j] = fun_Ctx(S[i][j], T, K, r, Sig[i][j])
            P[i][j] = fun_Ptx(S[i][j], T, K, r, Sig[i][j])
    surfC = ax.plot_surface(Sig, S, C, color='b', label='Call Option')
    surfC._facecolors2d=surfC._facecolor3d
    surfC._edgecolors2d=surfC._edgecolor3d
    surfP = ax.plot_surface(Sig, S, P, color='g', label='Put Option')
    surfP._facecolors2d=surfP._facecolor3d
    surfP._edgecolors2d=surfP._edgecolor3d
    ax.legend()
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Initial Stock Price')
    ax.set_zlabel('Option Price')
    plt.title('C(sig,x) && P(sig,x) v/s x && sig')
    plt.show()
    print("")

def Q4_varyT():
    print("Varying Time");
    print("Part I <-> 2D Plot")
    S = np.linspace(0.8, 1.2, 5)
    T = np.linspace(0.1,0.99, 50)
    for s in S:
        C = [fun_Ctx(s, t, K, r, sig) for t in T]
        plt.plot(T, C, label='S = {}'.format(s))
    plt.xlabel('Time')
    plt.ylabel('Call Option Price')
    plt.title('C(t) v/s t')
    plt.legend()
    plt.show()
    plt.clf()

    for s in S:
        P = [fun_Ptx(s, t, K, r, sig) for t in T]
        plt.plot(T, P, label='S = {}'.format(s))
    plt.xlabel('Time')
    plt.ylabel('Put Option Price')
    plt.title('P(t) v/s t')
    plt.legend()
    plt.show()

    print("Part II <-> Table")
    print("")
    S = 0.8
    T = np.linspace(0.1, 0.9, 5)
    C = [fun_Ctx(S, t, K, r, sig) for t in T]
    P = [fun_Ptx(S, t, K, r, sig) for t in T]
    table = []
    for i in range(len(T)):
        table.append([T[i], C[i], P[i]])
    df = pd.DataFrame(table, columns = ['Time','Call Price','Put Price'])
    print(df)
    print("")

    print("Part III <-> 3D Plot")
    T = np.linspace(0.1, 0.99, 25)
    S = np.linspace(0.8, 1.2, 25)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    T, S = np.meshgrid(T, S)
    C = np.zeros_like(T)
    P = np.zeros_like(T)
    for i in range(25):
        for j in range(25):
            C[i][j] = fun_Ctx(S[i][j], T[i][j], K, r, sig)
            P[i][j] = fun_Ptx(S[i][j], T[i][j], K, r, sig)
    surfC = ax.plot_surface(T, S, C, color='b', label='Call Option')
    surfC._facecolors2d=surfC._facecolor3d
    surfC._edgecolors2d=surfC._edgecolor3d
    surfP = ax.plot_surface(T, S, P, color='g', label='Put Option')
    surfP._facecolors2d=surfP._facecolor3d
    surfP._edgecolors2d=surfP._edgecolor3d
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Initial Stock Price')
    ax.set_zlabel('Option Price')
    plt.title('C(t,x) && P(t,x) v/s x && t')
    plt.show()
    print("")

Q4_varyK()
Q4_varyR()
Q4_varySigma()
Q4_varyT()

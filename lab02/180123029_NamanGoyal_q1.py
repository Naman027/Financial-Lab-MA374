import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math
  
def factorial(n): 
    res = 1
    for i in range(2, n+1): 
        res = res * i 
    return res 

def solve1(S0,K,T,M,r,Sigma):
    t = (T*1.0)/(M*1.0)
    u = math.exp((Sigma*math.sqrt(t))) 
    d = math.exp(-1*(Sigma*math.sqrt(t)))
    p = (math.exp(r*t)-d)/(u-d)
    q = 1-p
    EC_Price = 0
    EP_Price = 0
    for k in range(0, M+1):
        S = S0*math.pow(u,k)*math.pow(d,M-k)
        S1 = max(0, S-K)
        S1 = S1*math.pow(p,k)*math.pow(q,M-k)
        S1 = S1*factorial(M)/(factorial(k)*factorial(M-k))
        EC_Price= EC_Price+S1
        S2 = max(0, K-S)
        S2 = S2*math.pow(p,k)*math.pow(q,M-k)
        S2 = S2*factorial(M)/(factorial(k)*factorial(M-k))
        EP_Price= EP_Price+S2
                
    EC_Price = EC_Price/math.pow(math.exp(r*t),M)
    EP_Price = EP_Price/math.pow(math.exp(r*t),M)
    return [EC_Price,EP_Price]

S0 = 100
K = 100
T = 1
r = 8.00/100.00
Sigma = 20.00/100.00
M = 100

P = solve1(S0,K,T,M,r,Sigma)
print("Vaue of European Call Option for set 1: %f"%(P[0]))
print("Vaue of European Put Option for set 1: %f"%(P[1]))
print("")

EC_Pr= []
EP_Pr= []
S0_G= []
for i in range(51, 151):
    P = solve1(i,K,T,M,r,Sigma)
    EC_Pr.append(P[0])
    EP_Pr.append(P[1])
    S0_G.append(i)
plt.plot(S0_G, EC_Pr, label="European Call Price")
plt.plot(S0_G, EP_Pr, label="European Put Price")
plt.xlabel("S0", size=20)
plt.ylabel("Price", size=20)
plt.title("Option Price Varying S0",size= 20)
plt.legend()
plt.show()

EC_Pr= []
EP_Pr= []
K_G= []
for k_i in range(51, 151):
    P= solve1(S0,k_i,T,M,r,Sigma)
    EC_Pr.append(P[0])
    EP_Pr.append(P[1])
    K_G.append(k_i)
plt.plot(K_G, EC_Pr, label="European Call Price")
plt.plot(K_G, EP_Pr, label="European Put Price")
plt.xlabel("K", size=20)
plt.ylabel("Price", size=20)
plt.title("Option Price Varying K",size= 20)
plt.legend()
plt.show()

EC_Pr= []
EP_Pr= []
Graph_r= []
for r_i in range(500, 1000, 5):
    P= solve1(S0,K,T,M,r_i/10000.00,Sigma)
    EC_Pr.append(P[0])
    EP_Pr.append(P[1])
    Graph_r.append(r_i/10000.00)
plt.plot(Graph_r, EC_Pr, label="European Call Price")
plt.plot(Graph_r, EP_Pr, label="European Put Price")
plt.xlabel("r", size=20)
plt.ylabel("Price", size=20)
plt.title("Option Price Varying r",size= 20)
plt.legend()
plt.show()

EC_Pr= []
EP_Pr= []
Sigma_G= []
for Sigm_i in range(1500, 2500, 5):
    P= solve1(S0,K,T,M,r,Sigm_i/10000.00)
    EC_Pr.append(P[0])
    EP_Pr.append(P[1])
    Sigma_G.append(Sigm_i/10000.00)
plt.plot(Sigma_G, EC_Pr, label="European Call Price")
plt.plot(Sigma_G, EP_Pr, label="European Put Price")
plt.xlabel("Sigma", size=20)
plt.ylabel("Price", size=20)
plt.title("Option Price Varying Sigma",size= 20)
plt.legend()
plt.show()

EC_Pr_95= []
EP_Pr_95= []
EC_Pr_100= []
EP_Pr_100= []
EC_Pr_105= []
EP_Pr_105= []
M_G= []
for m_i in range(51, 151):
    P= solve1(S0,95,T,m_i,r,Sigma)
    EC_Pr_95.append(P[0])
    EP_Pr_95.append(P[1])
    
    P= solve1(S0,100,T,m_i,r,Sigma)
    EC_Pr_100.append(P[0])
    EP_Pr_100.append(P[1])
    
    P= solve1(S0,105,T,m_i,r,Sigma)
    EC_Pr_105.append(P[0])
    EP_Pr_105.append(P[1])
    
    M_G.append(m_i)

plt.plot(M_G, EC_Pr_95, label="European Call Price for k = 95")
plt.plot(M_G, EP_Pr_95, label="European Put Price for k = 95")
plt.plot(M_G, EC_Pr_100, label="European Call Price for k = 100")
plt.plot(M_G, EP_Pr_100, label="European Put Price for k = 100")
plt.plot(M_G, EC_Pr_105, label="European Call Price k = 105")
plt.plot(M_G, EP_Pr_105, label="European Put Price k = 105")
plt.xlabel('M', size=20)
plt.ylabel('Price', size=20)
plt.title("Option price varying M for k = 95, 100, 105 ",size= 20)
plt.legend()
plt.show()

def plotBigGraph(X, Y, Z1, Z2, XLabel, YLabel, G1, G2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.array(X), np.array(Y), np.array(Z1),linewidth=0)
    ax.set_xlabel(XLabel, size=20)
    ax.set_ylabel(YLabel, size=20)
    ax.set_zlabel('Price', size=20)
    plt.title(G1, size= 20)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(np.array(X), np.array(Y), np.array(Z2),linewidth=0)
    ax.set_xlabel(XLabel, size=20)
    ax.set_ylabel(YLabel, size=20)
    ax.set_zlabel('Price', size=20)
    plt.title(G2, size= 20)
    plt.show()

S0_G= []
K_G= []
EC_Pr= []
EP_Pr= []
for i in range(90, 110):
    a= []
    b= []
    c= []
    d= []
    for k_i in range(90, 110):
        a.append(i)
        b.append(k_i)
        P = solve1(i,k_i,T,M,r,Sigma)
        c.append(P[0])
        d.append(P[1])
    S0_G.append(a)
    K_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(S0_G, K_G, EC_Pr, EP_Pr, "S", "K", "Call Price varying S,k", "Put Price varying S,k")

S0_G= []
M_G= []
EC_Pr= []
EP_Pr= []
for i in range(90, 110):
    a= []
    b= []
    c= []
    d= []
    for m_i in range(90, 110):
        a.append(i)
        b.append(m_i)
        P= solve1(i,K,T,m_i,r,Sigma)
        c.append(P[0])
        d.append(P[1])
    S0_G.append(a)
    M_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(S0_G, M_G, EC_Pr, EP_Pr, "S", "M", "Call Price varying S,M", "Put Price varying S,M")
    
K_G= []
M_G= []
EC_Pr= []
EP_Pr= []
for k_i in range(90, 110):
    a= []
    b= []
    c= []
    d= []
    for m_i in range(90, 110):
        a.append(k_i)
        b.append(m_i)
        P= solve1(S0,k_i,T,m_i,r,Sigma)
        c.append(P[0])
        d.append(P[1])
    K_G.append(a)
    M_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(K_G, M_G, EC_Pr, EP_Pr, "K", "M", "Call Price varying K,M", "Put Price varying K,M")

Graph_r= []
Sigma_G= []
EC_Pr= []
EP_Pr= []
for r_i in range(500, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for Sigm_i in range(1700, 2200, 10):
        a.append(r_i/10000.0)
        b.append(Sigm_i/10000.0)
        P= solve1(S0,K,T,M,r_i/10000.0,Sigm_i/10000.0)
        c.append(P[0])
        d.append(P[1])
    Graph_r.append(a)
    Sigma_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Graph_r, Sigma_G, EC_Pr, EP_Pr, "r", "Sigma", "Call Price varying r,Sigma", "Put Price varying r,Sigma")

Graph_r= []
S0_G= []
EC_Pr= []
EP_Pr= []
for r_i in range(500, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for i in range(90, 110):
        a.append(r_i/10000.0)
        b.append(i)
        P= solve1(i,K,T,M,r_i/10000.0,Sigma)
        c.append(P[0])
        d.append(P[1])
    Graph_r.append(a)
    S0_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Graph_r, S0_G, EC_Pr, EP_Pr, "r", "S0", "Call Price varying r,S0", "Put Price varying r,S0")

Graph_r= []
K_G= []
EC_Pr= []
EP_Pr= []
for r_i in range(500, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for k_i in range(90, 110):
        a.append(r_i/10000.0)
        b.append(k_i)
        P= solve1(S0,k_i,T,M,r_i/10000.0,Sigma)
        c.append(P[0])
        d.append(P[1])
    Graph_r.append(a)
    K_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Graph_r, K_G, EC_Pr, EP_Pr, "r", "K", "Call Price varying r,K", "Put Price varying r,K")

Graph_r= []
M_G= []
EC_Pr= []
EP_Pr= []
for r_i in range(500, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for m_i in range(90, 110):
        a.append(r_i/10000.0)
        b.append(m_i)
        P= solve1(S0,K,T,m_i,r_i/10000.0,Sigma)
        c.append(P[0])
        d.append(P[1])
    Graph_r.append(a)
    M_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Graph_r, M_G, EC_Pr, EP_Pr, "r", "M", "Call Price varying r,M", "Put Price varying r,M")

Sigma_G= []
S0_G= []
EC_Pr= []
EP_Pr= []
for Sigm_i in range(1700, 2200, 10):
    a= []
    b= []
    c= []
    d= []
    for i in range(90, 110):
        a.append(Sigm_i/10000.0)
        b.append(i)
        P= solve1(i,K,T,M,r,Sigm_i/10000.0)
        c.append(P[0])
        d.append(P[1])
    Sigma_G.append(a)
    S0_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Sigma_G, S0_G, EC_Pr, EP_Pr, "Sigma", "S0", "Call Price varying Sigma,S0", "Put Price varying Sigma,S0")

Sigma_G= []
K_G= []
EC_Pr= []
EP_Pr= []
for Sigm_i in range(1700, 2200, 10):
    a= []
    b= []
    c= []
    d= []
    for k_i in range(90, 110):
        a.append(Sigm_i/10000.0)
        b.append(k_i)
        P= solve1(S0,k_i,T,M,r,Sigm_i/10000.0)
        c.append(P[0])
        d.append(P[1])
    Sigma_G.append(a)
    K_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Sigma_G, K_G, EC_Pr, EP_Pr, "Sigma", "K", "Call Price varying Sigma,K", "Put Price varying Sigma,K")

Sigma_G= []
M_G= []
EC_Pr= []
EP_Pr= []
for Sigm_i in range(1700, 2200, 10):
    a= []
    b= []
    c= []
    d= []
    for m_i in range(90, 110):
        a.append(Sigm_i/10000.0)
        b.append(m_i)
        P= solve1(S0,K,T,m_i,r,Sigm_i/10000.0)
        c.append(P[0])
        d.append(P[1])
    Sigma_G.append(a)
    M_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Sigma_G, M_G, EC_Pr, EP_Pr, "Sigma", "M", "Call Price varying Sigma,M", "Put Price varying Sigma,M")

def solve2(S0,K,T,M,r,Sigma):
    t = (T*1.0)/(M*1.0)
    u = math.exp((Sigma*math.sqrt(t)) + ((r-(Sigma*Sigma)/2.0)*t))
    d = math.exp(-1*(Sigma*math.sqrt(t)) + ((r-(Sigma*Sigma)/2.0)*t))
    p = (math.exp(r*t)-d)/(u-d)
    q = 1-p
    EC_Price = 0
    EP_Price = 0
    for k in range(0, M+1):
        S = S0*math.pow(u,k)*math.pow(d,M-k)
        S1 = max(0, S-K)
        S1 = S1*math.pow(p,k)*math.pow(q,M-k)
        S1 = S1*factorial(M)/(factorial(k)*factorial(M-k))
        EC_Price= EC_Price+S1

        S2 = max(0, K-S)
        S2 = S2*math.pow(p,k)*math.pow(q,M-k)
        S2 = S2*factorial(M)/(factorial(k)*factorial(M-k))
        EP_Price= EP_Price+S2
                
    EC_Price = EC_Price/math.pow(math.exp(r*t),M)
    EP_Price = EP_Price/math.pow(math.exp(r*t),M)
    return [EC_Price,EP_Price]

P = solve2(S0,K,T,M,r,Sigma)
print("Vaue of European Call Option for set 2: %f"%(P[0]))
print("Vaue of European Put Option for set 2: %f"%(P[1]))
print("")

EC_Pr= []
EP_Pr= []
S0_G= []
for i in range(51, 151):
    P = solve2(i,K,T,M,r,Sigma)
    EC_Pr.append(P[0])
    EP_Pr.append(P[1])
    S0_G.append(i)
plt.plot(S0_G, EC_Pr, label="European Call Price")
plt.plot(S0_G, EP_Pr, label="European Put Price")
plt.xlabel("S0", size=20)
plt.ylabel("Price", size=20)
plt.title("Option Price Varying S0",size= 20)
plt.legend()
plt.show()

EC_Pr= []
EP_Pr= []
K_G= []
for k_i in range(51, 151):
    P= solve2(S0,k_i,T,M,r,Sigma)
    EC_Pr.append(P[0])
    EP_Pr.append(P[1])
    K_G.append(k_i)
plt.plot(K_G, EC_Pr, label="European Call Price")
plt.plot(K_G, EP_Pr, label="European Put Price")
plt.xlabel("K", size=20)
plt.ylabel("Price", size=20)
plt.title("Option Price Varying K",size= 20)
plt.legend()
plt.show()

EC_Pr= []
EP_Pr= []
Graph_r= []
for r_i in range(500, 1000, 5):
    P= solve2(S0,K,T,M,r_i/10000.00,Sigma)
    EC_Pr.append(P[0])
    EP_Pr.append(P[1])
    Graph_r.append(r_i/10000.00)
plt.plot(Graph_r, EC_Pr, label="European Call Price")
plt.plot(Graph_r, EP_Pr, label="European Put Price")
plt.xlabel("r", size=20)
plt.ylabel("Price", size=20)
plt.title("Option Price Varying r",size= 20)
plt.legend()
plt.show()

EC_Pr= []
EP_Pr= []
Sigma_G= []
for Sigm_i in range(1500, 2500, 5):
    P= solve2(S0,K,T,M,r,Sigm_i/10000.00)
    EC_Pr.append(P[0])
    EP_Pr.append(P[1])
    Sigma_G.append(Sigm_i/10000.00)
plt.plot(Sigma_G, EC_Pr, label="European Call Price")
plt.plot(Sigma_G, EP_Pr, label="European Put Price")
plt.xlabel("Sigma", size=20)
plt.ylabel("Price", size=20)
plt.title("Option Price Varying Sigma",size= 20)
plt.legend()
plt.show()

EC_Pr_95= []
EP_Pr_95= []
EC_Pr_100= []
EP_Pr_100= []
EC_Pr_105= []
EP_Pr_105= []
M_G= []
for m_i in range(51, 151):
    P= solve2(S0,95,T,m_i,r,Sigma)
    EC_Pr_95.append(P[0])
    EP_Pr_95.append(P[1])
    
    P= solve2(S0,100,T,m_i,r,Sigma)
    EC_Pr_100.append(P[0])
    EP_Pr_100.append(P[1])
    
    P= solve2(S0,105,T,m_i,r,Sigma)
    EC_Pr_105.append(P[0])
    EP_Pr_105.append(P[1])
    
    M_G.append(m_i)
plt.plot(M_G, EC_Pr_95, label="European Call Price for k = 95")
plt.plot(M_G, EP_Pr_95, label="European Put Price for k = 95")
plt.plot(M_G, EC_Pr_100, label="European Call Price for k = 100")
plt.plot(M_G, EP_Pr_100, label="European Put Price for k = 100")
plt.plot(M_G, EC_Pr_105, label="European Call Price k = 105")
plt.plot(M_G, EP_Pr_105, label="European Put Price k = 105")
plt.xlabel('M', size=20)
plt.ylabel('Price', size=20)
plt.title("Option price varying M for k = 95, 100, 105 ",size= 20)
plt.legend()
plt.show()

S0_G= []
K_G= []
EC_Pr= []
EP_Pr= []
for i in range(90, 110):
    a= []
    b= []
    c= []
    d= []
    for k_i in range(90, 110):
        a.append(i)
        b.append(k_i)
        P= solve2(i,k_i,T,M,r,Sigma)
        c.append(P[0])
        d.append(P[1])
    S0_G.append(a)
    K_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(S0_G, K_G, EC_Pr, EP_Pr, "S", "K", "Call Price varying S,K", "Put Price varying S,K")

S0_G= []
M_G= []
EC_Pr= []
EP_Pr= []
for i in range(90, 110):
    a= []
    b= []
    c= []
    d= []
    for m_i in range(90, 110):
        a.append(i)
        b.append(m_i)
        P= solve2(i,K,T,m_i,r,Sigma)
        c.append(P[0])
        d.append(P[1])
    S0_G.append(a)
    M_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(S0_G, M_G, EC_Pr, EP_Pr, "S", "M", "Call Price varying S,M", "Put Price varying S,M")
    
K_G= []
M_G= []
EC_Pr= []
EP_Pr= []
for k_i in range(90, 110):
    a= []
    b= []
    c= []
    d= []
    for m_i in range(90, 110):
        a.append(k_i)
        b.append(m_i)
        P= solve2(S0,k_i,T,m_i,r,Sigma)
        c.append(P[0])
        d.append(P[1])
    K_G.append(a)
    M_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(K_G, M_G, EC_Pr, EP_Pr, "K", "M", "Call Price varying K,M", "Put Price varying K,M")

Graph_r= []
Sigma_G= []
EC_Pr= []
EP_Pr= []
for r_i in range(500, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for Sigm_i in range(1700, 2200, 10):
        a.append(r_i/10000.0)
        b.append(Sigm_i/10000.0)
        P= solve2(S0,K,T,M,r_i/10000.0,Sigm_i/10000.0)
        c.append(P[0])
        d.append(P[1])
    Graph_r.append(a)
    Sigma_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Graph_r, Sigma_G, EC_Pr, EP_Pr, "r", "Sigma", "Call Price varying r,Sigma", "Put Price varying r,Sigma")

Graph_r= []
S0_G= []
EC_Pr= []
EP_Pr= []
for r_i in range(500, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for i in range(90, 110):
        a.append(r_i/10000.0)
        b.append(i)
        P= solve2(i,K,T,M,r_i/10000.0,Sigma)
        c.append(P[0])
        d.append(P[1])
    Graph_r.append(a)
    S0_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Graph_r, S0_G, EC_Pr, EP_Pr, "r", "S0", "Call Price varying r,S0", "Put Price varying r,S0")

Graph_r= []
K_G= []
EC_Pr= []
EP_Pr= []
for r_i in range(500, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for k_i in range(90, 110):
        a.append(r_i/10000.0)
        b.append(k_i)
        P= solve2(S0,k_i,T,M,r_i/10000.0,Sigma)
        c.append(P[0])
        d.append(P[1])
    Graph_r.append(a)
    K_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Graph_r, K_G, EC_Pr, EP_Pr, "r", "K", "Call Price varying r,K", "Put Price varying r,K")

Graph_r= []
M_G= []
EC_Pr= []
EP_Pr= []
for r_i in range(500, 1000, 10):
    a= []
    b= []
    c= []
    d= []
    for m_i in range(90, 110):
        a.append(r_i/10000.0)
        b.append(m_i)
        P= solve2(S0,K,T,m_i,r_i/10000.0,Sigma)
        c.append(P[0])
        d.append(P[1])
    Graph_r.append(a)
    M_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Graph_r, M_G, EC_Pr, EP_Pr, "r", "M", "Call Price varying r,M", "Put Price varying r,M")

Sigma_G= []
S0_G= []
EC_Pr= []
EP_Pr= []
for Sigm_i in range(1700, 2200, 10):
    a= []
    b= []
    c= []
    d= []
    for i in range(90, 110):
        a.append(Sigm_i/10000.0)
        b.append(i)
        P= solve2(i,K,T,M,r,Sigm_i/10000.0)
        c.append(P[0])
        d.append(P[1])
    Sigma_G.append(a)
    S0_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Sigma_G, S0_G, EC_Pr, EP_Pr, "Sigma", "S0", "Call Price varying Sigma,S0", "Put Price varying Sigma,S0")

Sigma_G= []
K_G= []
EC_Pr= []
EP_Pr= []
for Sigm_i in range(1700, 2200, 10):
    a= []
    b= []
    c= []
    d= []
    for k_i in range(90, 110):
        a.append(Sigm_i/10000.0)
        b.append(k_i)
        P= solve2(S0,k_i,T,M,r,Sigm_i/10000.0)
        c.append(P[0])
        d.append(P[1])
    Sigma_G.append(a)
    K_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Sigma_G, K_G, EC_Pr, EP_Pr, "Sigma", "K", "Call Price varying Sigma,K", "Put Price varying Sigma,K")

Sigma_G= []
M_G= []
EC_Pr= []
EP_Pr= []
for Sigm_i in range(1700, 2200, 10):
    a= []
    b= []
    c= []
    d= []
    for m_i in range(90, 110):
        a.append(Sigm_i/10000.0)
        b.append(m_i)
        P= solve2(S0,K,T,m_i,r,Sigm_i/10000.0)
        c.append(P[0])
        d.append(P[1])
    Sigma_G.append(a)
    M_G.append(b)
    EC_Pr.append(c)
    EP_Pr.append(d)
plotBigGraph(Sigma_G, M_G, EC_Pr, EP_Pr, "Sigma", "M", "Call Price varying Sigma,M", "Put Price varying Sigma,M")

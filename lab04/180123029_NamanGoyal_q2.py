from cvxopt import matrix
from cvxopt import solvers
solvers.options['show_progress'] = False
import numpy as np
import matplotlib.pyplot as plt
import math
import time

R=[]
for i in range(201):
  R.append(0.1+i*0.0005)
  
Sigma,w1,w2=[],[],[]
for i in range(len(R)):
  r=R[i]
  Q=matrix([[0.01, -0.02, 0.008],[-0.02, 0.080, -0.004],[0.008, -0.004, 0.046]])
  p=matrix([0.0, 0.0, 0.0])
  G=matrix([[-1.0, 0.0, 0.0],[0.0, -1.0, 0.0],[0.0, 0.0, -1.0]])
  h=matrix([-0.0, -0.0, -0.0])
  A=matrix([[ 0.1, 1.0],[0.2, 1.0],[0.15, 1.0]])
  b=matrix([r, 1.0])
  sol = solvers.qp(Q,p,G,h,A,b)
  sigm=math.sqrt(sol['primal objective'])
  w1.append(sol['x'][0])
  w2.append(sol['x'][1])
  Sigma.append(sigm)

Eff1,Eff2=[],[]
mmk=min(Sigma)
flag=0
for i in range(len(R)):
  if Sigma[i]==mmk:
    flag=i
for i in range(len(R)):
  if i>=flag:
    Eff1.append(R[i])
    Eff2.append(Sigma[i])

N = 141
R12=[]

def CalcR12():
  for j in range(141):
    R12.append(0.08+j*0.001)

CalcR12()
Sigma12=[]
Sigma23=[]
Sigma13=[]

def SigCalc():
  for i in range(N):
    r=R12[i]
    Q=matrix([[0.01, -0.02, 0.008],[-0.02, 0.080, -0.004],[0.008, -0.004, 0.046]])
    p=matrix([0.0, 0.0, 0.0])
    G=matrix([[-0.0, 0.0, 0.0],[0.0, -0.0, 0.0],[0.0, 0.0, -0.0]])
    h=matrix([-0.0, -0.0, -0.0])
    A=matrix([[ 0.1, 1.0, 0.0],[0.2, 1.0, 0.0],[0.15, 0.0, 1.0]])
    b=matrix([r, 1.0, 0.0])
    sol = solvers.qp(Q,p,G,h,A,b)
    sigm=math.sqrt(sol['primal objective'])
    Sigma12.append(sigm)
  
  for i in range(N):
    r=R12[i]
    Q=matrix([[0.01, -0.02, 0.008],[-0.02, 0.080, -0.004],[0.008, -0.004, 0.046]])
    p=matrix([0.0, 0.0, 0.0])
    G=matrix([[-0.0, 0.0, 0.0],[0.0, -0.0, 0.0],[0.0, 0.0, -0.0]])
    h=matrix([-0.0, -0.0, -0.0])
    A=matrix([[ 0.1, 1.0, 0.0],[0.2, 0.0, 1.0],[0.15, 1.0, 0.0]])
    b=matrix([r, 1.0, 0.0])
    sol = solvers.qp(Q,p,G,h,A,b)
    sigm=math.sqrt(sol['primal objective'])
    Sigma13.append(sigm)
  
  for i in range(N):
    r=R12[i]
    Q=matrix([[0.01, -0.02, 0.008],[-0.02, 0.080, -0.004],[0.008, -0.004, 0.046]])
    p=matrix([0.0, 0.0, 0.0])
    G=matrix([[-0.0, 0.0, 0.0],[0.0, -0.0, 0.0],[0.0, 0.0, -0.0]])
    h=matrix([-0.0, -0.0, -0.0])
    A=matrix([[ 0.1, 0.0, 1.0],[0.2, 1.0, 0.0],[0.15, 1.0, 0.0]])
    b=matrix([r, 1.0, 0.0])
    sol = solvers.qp(Q,p,G,h,A,b)
    sigm=math.sqrt(sol['primal objective'])
    Sigma23.append(sigm)
  
SigCalc()
A,B=[],[]

def calcIt():
  for i in range(20):
    a=0.05*i
    for j in range(21-i):
      A.append(a)
      B.append(0.05*(j))

calcIt()
c=[[0.005, -0.01, 0.004],[-0.01, 0.04, -0.002],[0.004, -0.002, 0.023]]
m=[0.1,0.2,0.15]
fin_Return,fin_Risk=[],[]
for j in range(len(A)):
  wcm=[A[j],B[j],1-(A[j]+B[j])]
  wcmm=np.transpose(wcm)
  fin_Return.append(wcm[0]*m[0]+wcm[1]*m[1]+wcm[2]*m[2])
  zk=np.dot(np.dot(wcm,c),wcmm)
  zk=math.sqrt(zk)
  fin_Risk.append(zk)

print('Plot for Feasible Region for No ShortSelling + Min Variance Line with 2 assets at a time')
print("--------------------------------------------------------------------")
print("")
plt.plot(Sigma12,R12,'g',label='MVL with asset 1,2')
plt.plot(Sigma13,R12,'y',label='MVL with asset 1,2')
plt.plot(Sigma23,R12,'b',label='MVL with asset 1,2')
plt.scatter(fin_Risk,fin_Return)
plt.xlabel('Risk')
plt.ylabel('Return')
plt.legend(loc='lower right')
plt.show()

print('Plot for Minimum Variance line with No ShortSelling + Minimum Variance Line with 2 assets at a time')
print("--------------------------------------------------------------------")
print("")
plt.plot(Sigma,R,'v',label='Minimum Variance Line (of 3 assets) with no shortselling')
plt.plot(Sigma12,R12,'g',label='MVL with asset 1,2')
plt.plot(Sigma13,R12,'y',label='MVL with asset 1,2')
plt.plot(Sigma23,R12,'b',label='MVL with asset 1,2')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.legend(loc='lower right')
plt.show()

print('Plot for Efficient Frontier with No ShortSelling + Minimum Variance Line with 2 assets at a time')
print("--------------------------------------------------------------------")
print("")
plt.plot(Eff2,Eff1,'v',label='Effecient frontier(of 3 assets) with No ShortSelling')
plt.plot(Sigma12,R12,'g',label='MVL with asset 1,2')
plt.plot(Sigma13,R12,'y',label='MVL with asset 1,2')
plt.plot(Sigma23,R12,'b',label='MVL with asset 1,2')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.legend(loc='lower right')
plt.show()

print('Plot for Relationship amongst the W1 and W2 for No ShortSelling Case')
print("--------------------------------------------------------------------")
print("")
plt.scatter(w1,w2)
plt.xlabel('W1')
plt.ylabel('W2')
plt.show()

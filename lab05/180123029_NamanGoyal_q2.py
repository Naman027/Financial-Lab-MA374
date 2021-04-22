import statistics
import math
import matplotlib.pyplot as plt
import pandas as pd

filepath1=r"nse_index.csv"
df2 = pd.read_csv(filepath1)
filepath2=r"bse_index.csv"
df1 = pd.read_csv(filepath2)
nse_ret,bse_ret=[],[]

for i in range(59):
  nse_ret.append( (df2.loc[i+1,'nse_index']-df2.loc[i,'nse_index'])/df2.loc[i,'nse_index'] )
  bse_ret.append( (df1.loc[i+1,'bse_index']-df1.loc[i,'bse_index'])/df1.loc[i,'bse_index'] )
var_nseret,var_bseret=statistics.variance(nse_ret),statistics.variance(bse_ret)
sd_nse,sd_bse=math.sqrt(var_nseret),math.sqrt(var_bseret)
nse_meanret=(sum(nse_ret)/59)*12
bse_meanret=(sum(bse_ret)/59)*12

def fun1():
  Type = 0 
  kim=['BSE','NSE','NON-INDEXED STOCKS']
  kkim=['bsedata1.csv','nsedata1.csv','nse_non_index_data1.csv']
  if Type==0:
    filepath=r"bsedata1.csv"
  if Type==1:
    filepath=r"nsedata1.csv"
  if Type==2:
    filepath=r"nse_non_index_data1.csv"
  print('Analysis of',kim[Type],'stocks ----------> ')
  defcon1 = pd.read_csv(filepath)
  stocks=[]
  ii=0
  for col in defcon1.columns:
    if ii>0:
      stocks.append(col)
    ii+=1
  ll=len(stocks)
  returnn,betaa=[],[]
  for i in range(ll):
    print("Stock :",stocks[i])
    temp,tempret=[],[]
    for j in range(60):
      temp.append( defcon1.loc[j,stocks[i]])
      if j>0:
        tempret.append( (temp[j]-temp[j-1])/temp[j-1] )
    
    if Type==1 or Type==2:
       aa=sum(nse_ret)/len(nse_ret)
    else :
       aa=sum(bse_ret)/len(bse_ret)
    bb,cov=sum(tempret)/len(tempret),0
    if Type==1 or Type==2:
      for k in range(len(tempret)):
         cov+=(tempret[k]-bb)*(nse_ret[k]-aa)
      cov=cov/(len(tempret))
      cov=cov/var_nseret
      beta=cov
      betaa.append(beta)
      st_ret=bb*12
      returnn.append(st_ret)
      res=st_ret-(0.1+(nse_meanret-0.1)*beta)
      if res>0:
        print("Stock undervalued")
      elif res<0:
        print("Stock overvalued")
      elif res==0:
        print("Stock lies on the Security Market Line")
    else :
      for k in range(len(tempret)):
         cov+=(tempret[k]-bb)*(bse_ret[k]-aa)
      cov=cov/(len(tempret))
      cov=cov/var_bseret
      beta=cov
      st_ret=bb*12
      betaa.append(beta)
      returnn.append(st_ret)
      res=st_ret-(0.1+(bse_meanret-0.1)*beta)
      if res>0:
        print("Stock undervalued")
      elif res<0:
        print("Stock overvalued")
      elif res==0:
        print("Stock lies on the Security Market Line")
  print("")
  print('Security Market Line')
  bet,rett=[],[]
  if Type==1 or Type==2:
    mark=nse_meanret
    mag='NSE'
  else:
    mark=bse_meanret
    mag='BSE'
  for hg in range(80):
    bet.append(0.05*hg)
    rett.append(0.1+bet[hg]*(mark-0.1))   
  plt.plot(bet,rett)
  plt.scatter(betaa,returnn) 
  for hj in range(10):
    plt.annotate(stocks[hj],(betaa[hj],returnn[hj]))
  
  plt.xlabel('beta')
  plt.ylabel('Return')
  plt.show()
  print("Chosen Market portfolio is :",mag)
  print("---------------------------------------------------------------------------------")

def fun2():
  Type = 1 
  kim=['BSE','NSE','NON-INDEXED STOCKS']
  kkim=['bsedata1.csv','nsedata1.csv','nse_non_index_data1.csv']
  if Type==0:
    filepath=r"bsedata1.csv"
  if Type==1:
    filepath=r"nsedata1.csv"
  if Type==2:
    filepath=r"nse_non_index_data1.csv"
  print('Analysis of',kim[Type],'stocks ----------> ')
  defcon1 = pd.read_csv(filepath)
  stocks=[]
  ii=0
  for col in defcon1.columns:
    if ii>0:
      stocks.append(col)
    ii+=1
  ll=len(stocks)
  returnn,betaa=[],[]
  for i in range(ll):
    print("Stock :",stocks[i])
    temp,tempret=[],[]
    for j in range(60):
      temp.append( defcon1.loc[j,stocks[i]])
      if j>0:
        tempret.append( (temp[j]-temp[j-1])/temp[j-1] )
    
    if Type==1 or Type==2:
       aa=sum(nse_ret)/len(nse_ret)
    else :
       aa=sum(bse_ret)/len(bse_ret)
    bb,cov=sum(tempret)/len(tempret),0
    if Type==1 or Type==2:
      for k in range(len(tempret)):
         cov+=(tempret[k]-bb)*(nse_ret[k]-aa)
      cov=cov/(len(tempret))
      cov=cov/var_nseret
      beta=cov
      betaa.append(beta)
      st_ret=bb*12
      returnn.append(st_ret)
      res=st_ret-(0.1+(nse_meanret-0.1)*beta)
      if res>0:
        print("Stock undervalued")
      elif res<0:
        print("Stock overvalued")
      elif res==0:
        print("Stock lies on the Security Market Line")
    else :
      for k in range(len(tempret)):
         cov+=(tempret[k]-bb)*(bse_ret[k]-aa)
      cov=cov/(len(tempret))
      cov=cov/var_bseret
      beta=cov
      st_ret=bb*12
      betaa.append(beta)
      returnn.append(st_ret)
      res=st_ret-(0.1+(bse_meanret-0.1)*beta)
      if res>0:
        print("Stock undervalued")
      elif res<0:
        print("Stock overvalued")
      elif res==0:
        print("Stock lies on the Security Market Line")
  print("")
  print('Security Market Line')
  bet,rett=[],[]
  if Type==1 or Type==2:
    mark=nse_meanret
    mag='NSE'
  else:
    mark=bse_meanret
    mag='BSE'
  for hg in range(80):
    bet.append(0.05*hg)
    rett.append(0.1+bet[hg]*(mark-0.1))   
  plt.plot(bet,rett)
  plt.scatter(betaa,returnn) 
  for hj in range(10):
    plt.annotate(stocks[hj],(betaa[hj],returnn[hj]))
  
  plt.xlabel('beta')
  plt.ylabel('Return')
  plt.show()
  print("Chosen Market portfolio is :",mag)
  print("---------------------------------------------------------------------------------")
  
def fun3():
  Type = 2 
  kim=['BSE','NSE','NON-INDEXED STOCKS']
  kkim=['bsedata1.csv','nsedata1.csv','nse_non_index_data1.csv']
  if Type==0:
    filepath=r"bsedata1.csv"
  if Type==1:
    filepath=r"nsedata1.csv"
  if Type==2:
    filepath=r"nse_non_index_data1.csv"
  print('Analysis of',kim[Type],'stocks ----------> ')
  defcon1 = pd.read_csv(filepath)
  stocks=[]
  ii=0
  for col in defcon1.columns:
    if ii>0:
      stocks.append(col)
    ii+=1
  ll=len(stocks)
  returnn,betaa=[],[]
  for i in range(ll):
    print("Stock :",stocks[i])
    temp,tempret=[],[]
    for j in range(60):
      temp.append( defcon1.loc[j,stocks[i]])
      if j>0:
        tempret.append( (temp[j]-temp[j-1])/temp[j-1] )
    
    if Type==1 or Type==2:
       aa=sum(nse_ret)/len(nse_ret)
    else :
       aa=sum(bse_ret)/len(bse_ret)
    bb,cov=sum(tempret)/len(tempret),0
    if Type==1 or Type==2:
      for k in range(len(tempret)):
         cov+=(tempret[k]-bb)*(nse_ret[k]-aa)
      cov=cov/(len(tempret))
      cov=cov/var_nseret
      beta=cov
      betaa.append(beta)
      st_ret=bb*12
      returnn.append(st_ret)
      res=st_ret-(0.1+(nse_meanret-0.1)*beta)
      if res>0:
        print("Stock undervalued")
      elif res<0:
        print("Stock overvalued")
      elif res==0:
        print("Stock lies on the Security Market Line")
    else :
      for k in range(len(tempret)):
         cov+=(tempret[k]-bb)*(bse_ret[k]-aa)
      cov=cov/(len(tempret))
      cov=cov/var_bseret
      beta=cov
      st_ret=bb*12
      betaa.append(beta)
      returnn.append(st_ret)
      res=st_ret-(0.1+(bse_meanret-0.1)*beta)
      if res>0:
        print("Stock undervalued")
      elif res<0:
        print("Stock overvalued")
      elif res==0:
        print("Stock lies on the Security Market Line")
  print("")
  print('Security Market Line')
  bet,rett=[],[]
  if Type==1 or Type==2:
    mark=nse_meanret
    mag='NSE'
  else:
    mark=bse_meanret
    mag='BSE'
  for hg in range(80):
    bet.append(0.05*hg)
    rett.append(0.1+bet[hg]*(mark-0.1))   
  plt.plot(bet,rett)
  plt.scatter(betaa,returnn) 
  for hj in range(10):
    plt.annotate(stocks[hj],(betaa[hj],returnn[hj]))
  
  plt.xlabel('beta')
  plt.ylabel('Return')
  plt.show()
  print("Chosen Market portfolio is :",mag)
  print("---------------------------------------------------------------------------------")

fun1()
fun2()
fun3()




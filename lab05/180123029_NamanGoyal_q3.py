import statistics
import numpy as np
import pandas as pd

filepath=r"nse_index.csv"
df2 = pd.read_csv(filepath)
filepath2=r"bse_index.csv"
df1 = pd.read_csv(filepath2)
nse_ret,bse_ret=[],[]

for i in range(59):
  nse_ret.append( (df2.loc[i+1,'nse_index']-df2.loc[i,'nse_index'])/df2.loc[i,'nse_index'] )
  bse_ret.append( (df1.loc[i+1,'bse_index']-df1.loc[i,'bse_index'])/df1.loc[i,'bse_index'] )
var_nseret,var_bseret=statistics.variance(nse_ret),statistics.variance(bse_ret)

def fun1(): 
  Type = 0
  kim=['BSE','NSE','NON-INDEXED STOCKS']
  kkim=['bsedata1.csv','nsedata1.csv','nse_non_index_data1.csv']
  if Type==0:
    filepathn=r"bsedata1.csv"
  if Type==1:
    filepathn=r"nsedata1.csv"
  if Type==2:
    filepathn=r"nse_non_index_data1.csv"
  print('Beta for',kim[Type],'stocks ----------> ')
  defcon1 = pd.read_csv(filepathn)
  stocks=[]
  ii=0
  for col in defcon1.columns:
    if ii>0:
      stocks.append(col)
    ii+=1
  ll=len(stocks)
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
      print("Beta Value for the stock is :", cov/var_nseret)
    else :
      for k in range(len(tempret)):
         cov+=(tempret[k]-bb)*(bse_ret[k]-aa)
      cov=cov/(len(tempret))
      print("Beta Value for the stock is :", cov/var_bseret)
  print("---------------------------------------------------------------------------------")

def fun2(): 
  Type = 1
  kim=['BSE','NSE','NON-INDEXED STOCKS']
  kkim=['bsedata1.csv','nsedata1.csv','nse_non_index_data1.csv']
  if Type==0:
    filepathn=r"bsedata1.csv"
  if Type==1:
    filepathn=r"nsedata1.csv"
  if Type==2:
    filepathn=r"nse_non_index_data1.csv"
  print('Beta for',kim[Type],'stocks ----------> ')
  defcon1 = pd.read_csv(filepathn)
  stocks=[]
  ii=0
  for col in defcon1.columns:
    if ii>0:
      stocks.append(col)
    ii+=1
  ll=len(stocks)
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
      print("Beta Value for the stock is :", cov/var_nseret)
    else :
      for k in range(len(tempret)):
         cov+=(tempret[k]-bb)*(bse_ret[k]-aa)
      cov=cov/(len(tempret))
      print("Beta Value for the stock is :", cov/var_bseret)
  print("---------------------------------------------------------------------------------")
  
def fun3(): 
  Type = 2
  kim=['BSE','NSE','NON-INDEXED STOCKS']
  kkim=['bsedata1.csv','nsedata1.csv','nse_non_index_data1.csv']
  if Type==0:
    filepathn=r"bsedata1.csv"
  if Type==1:
    filepathn=r"nsedata1.csv"
  if Type==2:
    filepathn=r"nse_non_index_data1.csv"
  print('Beta for',kim[Type],'stocks ----------> ')
  defcon1 = pd.read_csv(filepathn)
  stocks=[]
  ii=0
  for col in defcon1.columns:
    if ii>0:
      stocks.append(col)
    ii+=1
  ll=len(stocks)
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
      print("Beta Value for the stock is :", cov/var_nseret)
    else :
      for k in range(len(tempret)):
         cov+=(tempret[k]-bb)*(bse_ret[k]-aa)
      cov=cov/(len(tempret))
      print("Beta Value for the stock is :", cov/var_bseret)
  print("---------------------------------------------------------------------------------")

fun1()
fun2()
fun3()



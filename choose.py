#!/usr/bin/env python
#coding=utf-8
"""
Create on Mon Oct 30 19:14:25 2017
Modify on Thu Nov 30 19:43:59 2017
@author:fly
Pretreatment stock data.
"""
import os
import csv
import numpy as np
import warnings as wn

def replace(series):
    temp=[]
    for i in range(len(series)-1,-1,-1):    #range(x,y,t) get number [x,y) by step t.
	temp.append(float(series[i]))
    return temp

def getcoldata(reader,col3,col5,col7):
    data3=[]
    data5=[]
    data7=[]
    for row in reader:
	data3.append(row[col3])
	data5.append(row[col5])
	data7.append(row[col7])
    return replace(data3[1:101]),replace(data5[1:101]),replace(data7[1:101])

wn.filterwarnings("ignore")
mydir='/home/fly/hs/data'
datadir='/home/fly/hs/interdata'
filelist=os.listdir(mydir)
change=[]
close=[]
trade=[]
for f in range(0,len(filelist)):
    path=os.path.join(mydir,filelist[f])
    (shortname,extension)=os.path.splitext(filelist[f])    #get shortname and extension from filename
    with open(path)as csvf:
	reader=csv.reader(csvf)
	tclose,ttrade,tchange=getcoldata(reader,3,5,7)
    change.append(tchange)
    close.append(tclose)
    trade.append(ttrade)
    print(shortname+" success!")
    csvf.close()
    #break

changepath=os.path.join(datadir,"change.csv")
closepath=os.path.join(datadir,"close.csv")
tradepath=os.path.join(datadir,"trade.csv")
np.savetxt(changepath,change,delimiter=',')  #save matrix to csv file
np.savetxt(closepath,close,delimiter=',')
np.savetxt(tradepath,trade,delimiter=',')

print("success") 

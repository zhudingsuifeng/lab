#!/usr/bin/env python
#coding=utf-8
"""
Create on Mon Oct 30 21:46:32 2017
Modify on Wed Dec 20 16:31:19 2017
@author:fly
Get stock close,calculate logarighmic price earnings and the pearson correlation coefficient of it and save as csv file.
"""
import os
import csv
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Reverse the sequence
def replace(series):
    temp=[]
    for i in range(len(series)-1,-1,-1):
	temp.append(float(series[i]))
    return temp

#get column data from specify directory file.
def getcoldata(datadir,name,col):
    data=[]
    path=os.path.join(datadir,name)
    with open(path) as csvf:
	reader=csv.reader(csvf)
	for row in reader:
	    data.append(row[col])
    csvf.close()
    return replace(data[1:101])

#calculate the logarighmic price earnings of stock.
def logearn(stockprice):
    #stockprice is sequence
    log=[]
    for i in range(0,len(stockprice)-1):
	earn=math.log(stockprice[i+1])-math.log(stockprice[i])
	log.append(earn)
    return log

#save data to csv file.
def savefile(savedir,filename,data):
    path=os.path.join(savedir,filename)
    np.savetxt(path,data,delimiter=',')


if __name__=="__main__":
    datadir='/home/fly/mygit/filter/stock'
    savedir='/home/fly/mygit/filter/result'
    filelist=os.listdir(datadir)   #list the file of the directory
    close=[]
    logs=[]

    #get stock close price.
    for i in range(0,len(filelist)):
	closecol=getcoldata(datadir,filelist[i],3)
	close.append(closecol)
    savefile(savedir,'close.csv',close)       #save stock close price to csv file
    print("get stock close success")

    #calculate the logarithmic price earnings.
    for i in range(0,len(close)):
	log=logearn(close[i])
	logs.append(log)
    savefile(savedir,'logearn.csv',logs)      #save logarithmic price earnings to csv file.
    print("calclate logearn success")

    #calculate the pearson correlation coefficients.
    pearson=np.corrcoef(logs)    #return Pearson correlation coefficients.
    for j in range(0,len(pearson)):
	for i in range(0,len(pearson)):
	    pearson[j][i]=("%.3f" % pearson[j][i]) #keep 3 decimal places.
    savefile(savedir,'pearson.csv',pearson)
    print("calculate pearson success")

    print("success")

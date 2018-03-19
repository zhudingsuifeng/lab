#!/usr/bin/env python
#coding=utf-8
"""
Create on Mon Oct 30 21:46:32 2017
Modify on Mon Mar 19 10:20:46 2018
@author:fly
Galculate logarighmic price earnings and the pearson correlation coefficient of it and save as csv file.
closefile:/home/fly/mygit/data/stock/close.csv
raw stock data:/home/fly/code/dataset
"""
import os
import csv
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#get stock code from stock data.
def getstockcode(datadir):
    stockcode=[]
    filelist=os.listdir(datadir)              #list the file of the directory
    for each in filelist:
	(code,ext)=os.path.splitext(each)     #separate file names and extensions
	stockcode.append(code)
    return stockcode

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
    data='/home/fly/mygit/data/stock/close.csv'
    stockdir='/home/fly/mygit/data/stock'
    savedir='/home/fly/mygit/data/similarity'
    logs=[]

    #get stock close price.
    close=np.loadtxt(data,delimiter=",")
    print("get stock close success")

    #calculate the logarithmic price earnings.
    for i in range(0,len(close)):
	log=logearn(close[i])
	logs.append(log)
    savefile(stockdir,'logearn.csv',logs)      #save logarithmic price earnings to csv file.
    print("calclate logearn success")

    #calculate the pearson correlation coefficients.
    pearson=np.corrcoef(logs)    #return Pearson correlation coefficients.
    '''
    for j in range(0,len(pearson)):
	for i in range(0,len(pearson)):
	    pearson[j][i]=("%.3f" % pearson[j][i]) #keep 3 decimal places.
    '''
    savefile(savedir,'pearson.csv',pearson)
    print("calculate pearson success")

    print("success")

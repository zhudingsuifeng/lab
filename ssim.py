#!/usr/bin/env python
#coding=utf-8
"""
Create on Tue Dec 12 10:51:01 2017
Modify on Tue Dec 19 19:43:22 2017
@author: fly
Find the high ssim value images.
opencv3.3.0
numpy1.13.1
pandas0.20.3
PIL4.2.1
skimage0.13.0
matplotlib(2.0.2)
"""
import os
import csv
import sys
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import warnings as wn
import cv2
import skimage.measure as sm
import networkx as nx
import pandas as pd
from PIL import Image
from compiler.ast import flatten

#get stock corresponding line
def getline(code,stocklist):
    for i in range(0,len(stocklist)):
	if code==stocklist[i]:      #stock name---------------------------------------------------
	    return i

if __name__=="__main__":
    wn.filterwarnings("ignore")
    print("Test case.")
    imgdir='/home/fly/mygit/filter/images'
    #datadir='/home/fly/mygit/filter/result/pearson.csv'
    datadir='/home/fly/mygit/filter/result/platessim.csv'
    #ssimdir='/home/fly/mygit/filter/sort/pearson'
    ssimdir='/home/fly/mygit/filter/sort/ssim'
    filelist=os.listdir(imgdir)          #list the file name in the dir

    #load ssim csv file
    ssim=np.loadtxt(open(datadir),delimiter=",",skiprows=0)
    print("load file success")

    #get stock code
    stockcode=[]
    for i in range(0,len(filelist)):
	(code,ext)=os.path.splitext(filelist[i])
	stockcode.append(code)
    print("get stock code")
	
    for row in range(0,len(ssim)):
	ssimline=ssim[row]
    	#merge the stock information.
    	stockinf=[]
    	#stockinf=ts.get_stock_basics()   #get stock data from tushare
    	with open("/home/fly/mygit/data/stock/industry.csv","r") as csvf: #open industry.csv file.
	    reader=csv.reader(csvf)
	    for line in reader:
	    	#merge the stock data.
	    	for i in range(0,len(stockcode)):
		    if stockcode[i]==line[1]:
		    	stockinf.append([line[1],line[2],ssimline[i],line[3]])
	    		#break
    	print("merge success")
     
    	#save data as csv file.
    	(shortname,ext)=os.path.splitext(filelist[row])   #save path--------------------------------------
    	path=os.path.join(ssimdir,shortname+'ssimsort.csv')
    	csvf=open(path,"w")   #open csv file
    	writer=csv.writer(csvf)
    	sortstock=sorted(stockinf,key=lambda x : x[2],reverse=True)    #sort the data
    	for i in range(0,60):   #60 high suitability stock
	    writer.writerow(sortstock[i])     #write data row to csv file
    	csvf.close()    #close csv file 
    	print("save "+str(row)+" file success")
    
    print("success")

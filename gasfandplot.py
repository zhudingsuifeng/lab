#!/usr/bin/env python
#coding=utf-8

'''
Created on Mon Oct 09:57:34 2017
Converts a time series into gasf image and draw series cruve into one image.
@author:fly
'''

import os
import csv
import networkx as nx
import warnings as wn
import matplotlib.pyplot as plt
import scipy.stats as stats 
import pandas as pd
import numpy as np

def rescale(series):
    maxval=max(series)
    minval=min(series)
    gap=float(maxval-minval)
    return [(each-minval)/gap for each in series]

def GASF(series):
    datacos=np.array(series)
    datasin=np.sqrt(1-np.array(series)**2)
    matcos=np.matrix(datacos)
    matsin=np.matrix(datasin)
    matrix=np.array(matcos.T*matcos-matsin.T*matsin)
    return matrix

wn.filterwarnings("ignore")
datapath='../data/changefile.csv'
imagedir='../images'

with open(datapath) as csvf:
    reader=csv.reader(csvf)                     #read file
    series=[]                                 #y
    stockcode=[]
    i=0
    for row in reader:
	series.append([])                     #create 2D array
	stockcode.append(row[0])                #get stockcode to seve file name
	for col in range(1,101):
	    series[i].append(float(row[col])) #get p change
	i+=1
    csvf.close()

for i in range(0,len(stockcode)):
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(1,2),dpi=300)   #figsize()specifies the aspect ratio,dpi specifies pixel resolution.
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0,hspace=0)
    zeroseries=rescale(series[i])
    gasfmatrix=GASF(zeroseries)
    ax1.axis('off')   #turn off the axis
    ax1.hlines(0,0,100,color='g',linewidth=0.2)  #draw a horizontal line y=0,x=0,100
    ax1.set_xlim(0,99)    #set the data limits for the x-axis.
    ax1.plot(series[i],color='red',linewidth=0.2)   #imput is series
    ax2.axis('off')
    ax2.imshow(gasfmatrix)     #imshow(),input is matrix
    imagepath=os.path.join(imagedir,stockcode[i]+'.png')
    plt.savefig(imagepath)
    plt.close()
    print(imagepath+'success!!!')
    
print("success")


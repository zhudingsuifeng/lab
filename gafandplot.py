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

def GADF(series):
    datacos=np.array(series)
    datasin=np.sqrt(1-np.array(series)**2)
    matcos=np.matrix(datacos)
    matsin=np.matrix(datasin)
    matrix=np.array(matsin.T*matcos-matcos.T*matsin)
    return matrix

wn.filterwarnings("ignore")
datapath='/home/fly/code/dataset'
filelist=os.listdir(datapath)
closepath='/home/fly/mygit/data/stock/close.csv'
tradepath='/home/fly/mygit/data/stock/trade.csv'
changepath='/home/fly/mygit/data/stock/change.csv'
gasfdir='/home/fly/mygit/images/trade/seriegasf'
gadfdir='/home/fly/mygit/images/trade/seriegadf'

raw=open(tradepath).readlines()     #get data from csv file
raw=[map(float,each.strip().split(',')) for each in raw]

for i in range(0,len(filelist)):
    (shortname,extension)=os.path.splitext(filelist[i])
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(1,2),dpi=300)   #figsize()specifies the aspect ratio,dpi specifies pixel resolution.
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0,hspace=0)
    #left,the left side of the subplots of the figure
    #right,the right side of the subplots of the figure
    #bottom,the bottom of the subplots of the figure
    #top,the top of the subplots of the figure
    #wspace,the amount of width reserved for blank space between subplots,expressed as a fraction of the average axis width
    #hspace,the amount of height reserved for white space between subplots,expressed as a fraction of the average axis heitht
    zeroseries=rescale(raw[i])
    #gasfmatrix=GASF(zeroseries)     #GASF
    gadfmatrix=GADF(zeroseries)     #GADF
    ax1.axis('off')   #turn off the axis
    ax1.hlines(0,0,100,color='g',linewidth=0.2)  #draw a horizontal line y=0,x=0,100
    ax1.set_xlim(0,99)    #set the data limits for the x-axis.
    ax1.plot(raw[i],color='red',linewidth=0.2)   #imput is series
    ax2.axis('off')
    ax2.imshow(gadfmatrix)     #imshow(),input is matrix
    imagepath=os.path.join(gadfdir,shortname+'.png')
    plt.savefig(imagepath)
    plt.close()
    print('save '+shortname+' success!!!')
    
print("success")


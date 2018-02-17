#!/usr/bin/env python
#coding=utf-8

'''
Created on Mon Oct 09:57:34 2017
Modfiy on Fri Dec 1 15:23:10 2017
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

def rescale(series,Type):
    maxval=max(series)
    minval=min(series)
    gap=float(maxval-minval)
    if Type=='Zero':
	return [(each-minval)/gap for each in series]
    if Type=='PI':
	return [((each-minval)/gap)*2*np.pi for each in series]
    else:
	sys.exit("Unknown rescaling type!")

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
polardir='/home/fly/mygit/images/polar'
rectangulardir='/home/fly/mygit/images/rectangular'

close=open(closepath).readlines()
close=[map(float,each.strip().split(',')) for each in close]  #G
trade=open(tradepath).readlines()
trade=[map(float,each.strip().split(',')) for each in trade]  #B
change=open(changepath).readlines()
change=[map(float,each.strip().split(',')) for each in change] #R

for i in range(0,len(filelist)):   #polar
    (shortname,extension)=os.path.splitext(filelist[i])
    fig,ax=plt.subplots(1,1,subplot_kw=dict(polar=True),figsize=(10,10),dpi=60)   #figsize()specifies the aspect ratio,dpi specifies pixel resolution.
    plt.title('Polar Coordinate',{'fontsize':20})
    zeroclose=rescale(close[i],'PI')    #G
    #zeroclose= map(np.arccos,zeroclose)
    zerotrade=rescale(trade[i],'PI')    #B
    #zerotrade= map(np.arccos,zerotrade)
    zerochange=rescale(change[i],'PI')  #R
    #zerochange=map(np.arccos,zerochange)
    r=np.arange(0,100,1)
    #t=np.linspace(0,2*np.pi,100)
    ax.plot(zeroclose,r,color='green',linewidth=2)   #imput is series
    ax.plot(zerotrade,r,color='blue',linewidth=2)
    ax.plot(zerochange,r,color='red',linewidth=2)
    imagepath=os.path.join(polardir,shortname+'.png')
    plt.savefig(imagepath)
    plt.close()
    print('save '+shortname+' success!!!')
    #break
    
for i in range(0,len(filelist)):   #rectangular
    (shortname,extension)=os.path.splitext(filelist[i])
    fig,ax=plt.subplots(1,1,figsize=(10,10),dpi=60)   #figsize()specifies the aspect ratio,dpi specifies pixel resolution.
    #plt.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0,hspace=0)
    plt.title('Time Series x',{'fontsize':20})   #title(label,fontdict) label is str ,fontdict is dict {key:value},a dictionary controlling the appearance of the title text.
    ax.set_xlim(0,99)    #set the data limits for the x-axis.
    zeroclose=rescale(close[i],'Zero')    #G
    zerotrade=rescale(trade[i],'Zero')    #B
    zerochange=rescale(change[i],'Zero')  #R
    ax.tick_params(axis='both',labelsize=15,direction='in',length=5,width=1,pad=2)
    #axis='both',axis on which to operate,labelsize ,tick label font size in points, direction,puts ticks inside the axes,outside the axes,or both,length,width,tick length and width in points,pad ,distance in points between tick and label.
    ax.plot(zeroclose,color='green',linewidth=2)   #imput is series
    ax.plot(zerotrade,color='blue',linewidth=2)
    ax.plot(zerochange,color='red',linewidth=2)
    imagepath=os.path.join(rectangulardir,shortname+'.png')
    plt.savefig(imagepath)
    plt.close()
    print('save '+shortname+' success!!!')
    #break

print("success")


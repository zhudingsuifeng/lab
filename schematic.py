#!/usr/bin/env python
#coding=utf-8

'''
Created on Sat Dec 2 20:26:53 2017
Plot serie to rectangular coordinates and polar coordinates,and convert serie to GAF image.
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
from mpl_toolkits.mplot3d import Axes3D

def rescale(series,Type):
    maxval=max(series)
    minval=min(series)
    gap=float(maxval-minval)
    if Type=='Zero':
	return [(each-minval)/gap for each in series]    #convert interval into (0,1)
    if Type=='PI':
	return [each*2*np.pi for each in series]         #convert interval into (0,2pi) 
    else:
	print("error")

def GASF(series):     #transform serie to GASF image
    datacos=np.array(series)
    datasin=np.sqrt(1-np.array(series)**2)
    matcos=np.matrix(datacos)
    matsin=np.matrix(datasin)
    matrix=np.array(matcos.T*matcos-matsin.T*matsin)
    return matrix

def GADF(series):     #transform serie to GADF image
    datacos=np.array(series)
    datasin=np.sqrt(1-np.array(series)**2)
    matcos=np.matrix(datacos)
    matsin=np.matrix(datasin)
    matrix=np.array(matsin.T*matcos-matcos.T*matsin)
    return matrix

#mixing r,g,b to rgb image
def mixing(r,g,b,length):
    rgb=np.zeros([length,length,3])
    for i in range(0,length):
	for k in range(0,length):
	    rgb[i][k][0]=r[i][k]
	    rgb[i][k][1]=g[i][k]
	    rgb[i][k][2]=b[i][k]
    return rgb

def savefig(path,name,image):
    savepath=os.path.join(path,name)
    plt.imsave(savepath,image,format="png")

wn.filterwarnings("ignore")
imagedir='/home/fly/mygit/images/schematic'
length=500    #set the image size
interval=np.arange(0,500,1)    #x
t=np.arange(0.0,7.5,0.015)     #
r=2*np.pi*t
g=2*np.pi*t+0.3*np.pi
b=2*np.pi*t+0.8*np.pi
R=np.sin(0.1*r)+np.sin(0.3*g)    #Fourier transform
R=rescale(R,'Zero')
G=np.sin(0.2*g)+np.sin(0.3*b)
G=rescale(G,'Zero')
B=np.sin(0.2*b)+np.sin(0.1*r)
B=rescale(B,'Zero')

RGASF=GASF(R)
GGASF=GASF(G)
BGASF=GASF(B)
GASF=mixing(RGASF,GGASF,BGASF,length)
savefig(imagedir,'RGASF.png',RGASF)
savefig(imagedir,'GGASF.png',GGASF)
savefig(imagedir,'BGASF.png',BGASF)
savefig(imagedir,'GASF.png',GASF)
print("GASF success")

RGADF=GADF(R)
GGADF=GADF(G)
BGADF=GADF(B)
GADF=mixing(RGADF,GGADF,BGADF,length)
savefig(imagedir,'RGADF.png',RGADF)
savefig(imagedir,'GGADF.png',GGADF)
savefig(imagedir,'BGADF.png',BGADF)
savefig(imagedir,'GADF.png',GADF)
print("GADF success")


for i in range(0,1):   #polar
    fig,ax=plt.subplots(1,1,subplot_kw=dict(polar=True),figsize=(10,10),dpi=60)   #figsize()specifies the aspect ratio,dpi specifies pixel resolution.
    plt.title('Polar Coordinate',{'fontsize':36})
    piG=rescale(G,'PI')
    piB=rescale(B,'PI')
    piR=rescale(R,'PI')
    ax.plot(piG,interval,color='green',linewidth=2)   #imput is series
    ax.plot(piB,interval,color='blue',linewidth=2)
    ax.plot(piR,interval,color='red',linewidth=2)
    imagepath=os.path.join(imagedir,'polar.png')
    plt.savefig(imagepath)
    plt.close()
    print('save polar success!!!')
    #break
    
for i in range(0,1):   #rectangular
    fig,ax=plt.subplots(1,1,figsize=(10,10),dpi=60)   #figsize()specifies the aspect ratio,dpi specifies pixel resolution.
    #plt.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0,hspace=0)
    plt.title('Time Series X1,X2,X3',{'fontsize':36})   #title(label,fontdict) label is str ,fontdict is dict {key:value},a dictionary controlling the appearance of the title text.
    ax.set_xlim(0,499)    #set the data limits for the x-axis.
    ax.tick_params(axis='both',labelsize=15,direction='in',length=5,width=1,pad=2)
    #axis='both',axis on which to operate,labelsize ,tick label font size in points, direction,puts ticks inside the axes,outside the axes,or both,length,width,tick length and width in points,pad ,distance in points between tick and label.
    ax.plot(G,color='green',linewidth=2)   #imput is series
    ax.plot(B,color='blue',linewidth=2)
    ax.plot(R,color='red',linewidth=2)
    imagepath=os.path.join(imagedir,'rectangular.png')
    plt.savefig(imagepath)
    plt.close()
    print('save rectangular success!!!')
    #break

print("success")


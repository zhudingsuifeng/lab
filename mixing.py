#!/usr/bin/env python
#coding=utf-8
"""
Created on Tue Aug 12 11:17:18 2014
Modified on Tue Sep 19 19:57:29 2017
@author: Fly
Convert times series to GASF,GADF,MTF images,and combine GASF,GADF and MTF to RGB images.
matplotlib(2.0.2)
networkx(2.0)
numpy(1.13.3)
opencv-python(3.3.0.10)
pandas(0.20.3)
Pillow(4.3.0)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import sys

#Rescale data into [0,1] or [-1,1] or original
def rescale(serie,Type):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    if Type=='Zero':       #rescale data into [0,1]
        return [(each-minval)/gap for each in serie]
    elif Type=='MaxZero': #rescale data into [-1,1]
	return [each/maxval for each in serie]
    elif Type=='None':     #return original data
	return serie    
    else:
	sys.exit("Unknown rescaling type!")

def GAF(series,Type):
    datacos=np.array(series)
    datasin=np.sqrt(1-np.array(series)**2)#np.array()create an array
    matcos=np.matrix(datacos)    #np.matrix(data),return a matrix from an array-like object.
    matsin=np.matrix(datasin)
    if Type=='GASF':
	matrix=np.array(matcos.T*matcos-matsin.T*matsin)
    elif Type=='GADF':
	matrix=np.array(matsin.T*matcos-matcos.T*matsin)
    else:
	sys.exit('Unknown type!')
    return matrix           #matrix is a 2-D array

#Calculate MTF matrix form time series (series) and quantitle bins(Q)
def MTF(series,Q,Type):
    Qcut=pd.qcut(series,Q,duplicates='drop') #pandas.qcut(),discretize variable into equal-sized buckets based on rank or based on sample quantiles."codes" contains bucket labels.(duplicates='drop'),drop the duplicate edges.
    label=Qcut.codes#the bins number
    MSM=np.zeros([Q,Q])
    matrix=[]
    patch=[]
    paamatrix=[]
    for i in range(0,len(label)-1):   #create the markov transition matrix
	MSM[label[i]][label[i+1]]+=1
    for i in xrange(Q):           
	if sum(MSM[i][:])==0:         #Prevent denominator from zero  
	    MSM[i][:]=1.0/Q
	     #continue
	MSM[i][:]=MSM[i][:]/sum(MSM[i][:])#Normalized the markov matrix
    for p in range(len(series)):      #create the MTF matrix
	for q in range(len(series)):
	    matrix.append(MSM[label[p]][label[q]]) #note!!!!!!
    if Type=='None':
	MTFmatrix=np.array(matrix).reshape(len(series),len(series))#array.reshape(l,l) gives a new shape to an array without changing its data.
    else:
	sys.exit('Unknown type!')
    return np.array(MTFmatrix)

def mixing(matrixa,matrixb,matrixc,length):
    matrix=np.zeros([length,length,3])
    for i in range(0,length):
	for k in range(0,length):
	    matrix[i][k][0]=matrixa[i][k]
	    matrix[i][k][1]=matrixb[i][k]
	    matrix[i][k][2]=matrixc[i][k]
    return matrix

def savefig(path,name,image):
    savepath=os.path.join(path,name+'.png')
    plt.imsave(savepath,image,format="png")   #imsave(data)required data is 2-D array

def loaddata(path):
    raw=open(path).readlines()
    raw=[map(float,each.strip().split(',')) for each in raw]
    return raw

#################################
###Define the parameters here####
#################################
if __name__=="__main__":
    Q=10
    length = 100
    mydir='/home/fly/code/dataset'
    filelist=os.listdir(mydir)
    close=loaddata('/home/fly/mygit/data/stock/close.csv')
    trade=loaddata('/home/fly/mygit/data/stock/trade.csv')
    change=loaddata('/home/fly/mygit/data/stock/change.csv')
            
    changegasf = []
    #changegadf = []
    closegasf = []
    #closegadf = []
    tradegasf = []
    #tradegadf = []
    gasfimage=[]
    #gadfimage=[]
    for each in close:
	closedata=rescale(each[:],'MaxZero')    #rescale def before
	GASFmatrix=GAF(closedata,'GASF')    #Without reduce the image size.GAFmatrix is a 2-D array.
	#GADFmatrix=GAF(closedata,'GADF')
	#MTFmatrix=MTF(each[:],Q,'None')
        closegasf.append(GASFmatrix)
	#closegadf.append(GADFmatrix)
	#closemtf.append(MTFmatrix)
    print("close success")
    
    for each in change:
	changedata=rescale(each[:],'Zero')
	GASFmatrix=GAF(changedata,'GASF')
	#GADFmatrix=GAF(changedata,'GADF')
	#MTFmatrix=MTF(each[:],Q,'None')
	changegasf.append(GASFmatrix)
	#changegadf.append(GADFmatrix)
	#changemtf.append(MTFmatrix)
    print("change success")

    for each in trade:
	tradedata=rescale(each[:],'MaxZero')
	GASFmatrix=GAF(tradedata,'GASF')
        #GADFmatrix=GAF(tradedata,'GADF')
        #MTFmatrix=MTF(each[:],Q,'None')
        tradegasf.append(GASFmatrix)
	#tradegadf.append(GADFmatrix)
	#trademtf.append(MTFmatrix)
    print("trade success")
    
    for i in range(0,len(close)):
	#gadfmatrix=mixing(changegadf[i],closegadf[i],tradegadf[i],length)
        #gadfimage.append(gadfmatrix)
	gasfmatrix=mixing(changegasf[i],closegasf[i],tradegasf[i],length)
        gasfimage.append(gasfmatrix)
	print("success +++")
    print("mixing success")

    #chtrclimage=np.asarray(chtrclimage)   #Convert the input to an array.Input should be the array_like.
    #gadfimage=np.asarray(gadfimage)
    gasfimage=np.asarray(gasfimage)

    for i in range(0,len(filelist)):
	(shortname,extension)=os.path.splitext(filelist[i])
	#savefig("/home/fly/mygit/images/mixing/chcltr/gadf",shortname,gadfimage[i])
	savefig("/home/fly/mygit/images/mixing/normal/rgbgasf",shortname,gasfimage[i])
	print("save "+shortname+" sucess!!!!")
    
    print("success")

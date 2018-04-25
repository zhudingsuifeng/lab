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

if __name__=="__main__":
    Q=10
    length = 100
    mydir='/home/fly/hs/data'
    filelist=os.listdir(mydir)
    close=loaddata('/home/fly/hs/interdata/close.csv')
    logearn=loaddata('/home/fly/hs/interdata/logearn.csv')
    turnover=loaddata('/home/fly/hs/interdata/turnover.csv')
    change=loaddata('/home/fly/hs/interdata/change.csv')
            
    changegasf = []
    logearngasf=[]
    closegasf = []
    turnovergasf = []
    gasfimage=[]
    for each in close:
	closedata=rescale(each[:],'Zero')    #rescale def before
	GASFmatrix=GAF(closedata,'GASF')    #Without reduce the image size.GAFmatrix is a 2-D array.
        closegasf.append(GASFmatrix)
    print("close success")
    
    for each in logearn:
	logearndata=rescale(each[:],'Zero')
	GASFmatrix=GAF(logearndata,'GASF')
	logearngasf.append(GASFmatrix)
    print("logearn success")
    
    for each in change:
	changedata=rescale(each[:],'Zero')
	GASFmatrix=GAF(changedata,'GASF')
	changegasf.append(GASFmatrix)
    print("change success")

    for each in turnover:
	turnoverdata=rescale(each[:],'Zero')
	GASFmatrix=GAF(turnoverdata,'GASF')
        turnovergasf.append(GASFmatrix)
    print("trade success")
    
    for i in range(0,len(change)):
	gasfmatrix=mixing(changegasf[i],logearngasf[i],turnovergasf[i],length)
        gasfimage.append(gasfmatrix)
	print("success +++")
    print("mixing success")

    gasfimage=np.asarray(gasfimage)

    for i in range(0,len(filelist)):
	(shortname,extension)=os.path.splitext(filelist[i])
	savefig("/home/fly/hs/lgasf",shortname,gasfimage[i])
	print("save "+shortname+" sucess!!!!")
    
    print("success")

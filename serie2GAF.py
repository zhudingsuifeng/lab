#!/usr/bin/env python
#coding=utf-8
"""
Created on Tue Aug 12 11:17:18 2014
Modified on Tue Sep 19 19:57:29 2017
@author: Fly
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
from mpl_toolkits.axes_grid1 import make_axes_locatable
import  sys

#Piecewise Aggregation Approximation(PAA) function
def paa(series, now, opw):
    if now == None:
        now = len(series) / opw
    if opw == None:
        opw = len(series) / now
    return [sum(series[i * opw : (i + 1) * opw]) / float(opw) for i in range(now)]

def standardize(serie):
    dev = np.sqrt(np.var(serie))
    mean = np.mean(serie)
    return [(each-mean)/dev for each in serie]

#Rescale data into [0,1] or [-1,1] or original
def rescale(serie,Type):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    if Type=='Zero':
        return [(each-minval)/gap for each in serie]
    elif Type=='Minusone':
	return [(each-minval)/gap*2-1 for each in serie]
    elif Type=='None':
	return serie 
    else:
	sys.exit("Unknown rescaling type!")

#get stock code from file "filename" and return stock code as a list.
def get_stock_code(filename):
    with open(filename) as csvf:
	stock_code=[]
	reader=csv.reader(csvf)
	for row in reader:
	    stock_code.append(row[0])
	return stock_code

#save image as "filename" from 2-D array,"array.ndim" can check the dimension of the array
def save_image(Type,filename,data):
    path=os.path.join('../',Type,filename,'.png')
    plt.imsave(path,data,format="png")

#Calculate GAF matrix form time series
def GAF(data,Type):
    datacos=np.array(data)
    datasin=np.sqrt(1-np.array(data)**2)#np.array()create an array
    matcos=np.matrix(datacos)    #np.matrix(data),return a matrix from an array-like object.
    matsin=np.matrix(datasin)
    if Type=='GASF':
	matrix=np.array(matcos.T*matcos-matsin.T*matsin)
    elif Type=='GADF':
	matrix=np.array(matsin.T*matcos-matcos.T*matsin)
    else:
	sys.exit('Unknown type!')
    return matrix

#
def QM(series,Q):
    print("success")

#
def MTF(data,Q):
    print("success")


#################################
###Define the parameters here####
#################################
if __name__=="__main__":
    #size = [64]  # PAA size
    size=64
    save_PAA = True # Save the GAF with or without dimension reduction by PAA: True, False
    rescale_type = 'Zero' # Rescale the data into [0,1] or [-1,1]: Zero, Minusone 
    datafile='../data/changefile.csv'
    #for s in size:  
    raw = open(datafile).readlines()
    raw = [map(float, each.strip().split(',')) for each in raw]
    length = len(raw[0])-1
            
    image = []
    paaimage = []
    patchimage = []
    for each in raw:
	Zerodata=rescale(each[1:],'Zero')
	Minusonedata=rescale(each[1:],'Minusone')
	Nonedata=rescale(each[1:],'None')

        paalistcos = paa(Zerodata,size,None) 
            
        ################raw################### 
	GASFmatrix=GAF(Zerodata,'GASF')    #Without reduce the image size.GAFmatrix is a 2-D array.
	GADFmatrix=GAF(Zerodata,'GADF')
	GASFpaamatrix=GAF(paalistcos,'GASF')#Reduce image size used Piecewise Aggregation Approximation(PAA)
	GADFpaamatrix=GAF(paalistcos,'GADF')
               
        image.append(GASFmatrix)       #image is a 3-D array.
	print(type(GASFmatrix),GASFmatrix.ndim,'!!!!')
        paaimage.append(GASFpaamatrix)
	print(type(GASFpaamatrix),GASFpaamatrix.ndim,'-------------')
    
    image = np.asarray(image)        #Convert the input to an array.Input should be the array_like.         
    paaimage = np.asarray(paaimage)
    print(type(image),image.ndim,'???????????????')
    '''
    ## draw large image and paa image
    k=0
    #plt.axis('off')
    with open("../data/changefile.csv") as csvf:
	reader=csv.reader(csvf)
	for row in reader:
	    path=os.path.join("../GASF",row[0],".png")
	    print(path,paaimage[k])
	    k+=1
            #plt.imsave("test.png",paaimage[k],format="png")#imsave(data)required data is 2-D array
	print(type(paaimage[0]),paaimage[0].ndim)
    '''
    print("success")

#!/usr/bin/env python
#coding=utf-8
"""
Created on Tue Aug 12 11:17:18 2014
Modified on Tue Sep 19 19:57:29 2017
@author: Fly
Convert times series to GASF,GADF,MTF images,and combine GASF,GADF and MTF to RGB images.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv
import sys

#Piecewise Aggregation Approximation(PAA) function.In fact,is a way to reduce the average dimension.
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
    if Type=='Zero':       #rescale data into [0,1]
        return [(each-minval)/gap for each in serie]
    elif Type=='Minusone': #rescale data into [-1,1]
	return [(each-minval)/gap*2-1 for each in serie]
    elif Type=='None':     #return original data
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
def MTF(series,Q,Type,size=64):
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
    elif Type=='PAA':
	paalist=paa(series,size,None)
	paaindex=[]
	for each in paalist:
	    for k in range(len(Qcut.categories)):
		if each>Qcut.categories[k].left and each <=Qcut.categories[k].right:#categories[k].left is lower bound ,right is upper bound.
		    paaindex.append(k)
	for p in range(size):
	    for q in range(size):
		paamatrix.append(MSM[paaindex[p]][paaindex[q]])
	MTFmatrix=np.array(paamatrix).reshape(size,size)
    elif Type=='PATCH':
	matrix=np.array(matrix).reshape(len(series),len(series))
	batch=len(series)/size
	for p in range(size):
	    for q in range(size):
		patch.append(np.mean(matrix[p*batch:(p+1)*batch,q*batch:(q+1)*batch]))
	MTFmatrix=np.array(patch).reshape(size,size)
    else:
	sys.exit('Unknown type!')
    return np.array(MTFmatrix)

#################################
###Define the parameters here####
#################################
if __name__=="__main__":
    Q=16
    size=64
    datafile='../data/changefile.csv'
    #for s in size:  
    raw = open(datafile).readlines()
    raw = [map(float, each.strip().split(',')) for each in raw]
    length = len(raw[0])-1
            
    GASFimage = []
    GADFimage = []
    MTFimage = []
    RGBimage = []
    paaimage = []
    patchimage = []
    for each in raw:
	Zerodata=rescale(each[1:],'Zero')
	Minusonedata=rescale(each[1:],'Minusone')
	Nonedata=rescale(each[1:],'None')

        paalistcos = paa(Zerodata,size,None) 
        paalist = paa(Nonedata,size,None);print("PAA")
    
        ################raw################### 
	GASFmatrix=GAF(Zerodata,'GASF')    #Without reduce the image size.GAFmatrix is a 2-D array.
	GADFmatrix=GAF(Zerodata,'GADF')
	GASFpaamatrix=GAF(paalistcos,'GASF')#Reduce image size used Piecewise Aggregation Approximation(PAA)
	GADFpaamatrix=GAF(paalistcos,'GADF');print("GAF")
        
	MTFmatrix=MTF(each[1:],Q,'None')
	MTFPAAmatrix=MTF(each[1:],Q,'PAA')
	MTFPATCHmatrix=MTF(each[1:],Q,'PATCH');print("MTF")
      
	RGBmatrix=np.zeros([length,length,3])
	for i in range(0,length):
	    for k in range(0,length):
		RGBmatrix[i][k][0]=GASFmatrix[i][k]
		RGBmatrix[i][k][1]=GADFmatrix[i][k]
		RGBmatrix[i][k][2]=MTFmatrix[i][k]
	print("RGB")
	
        GASFimage.append(GASFmatrix)       #image is a 3-D array.
	GADFimage.append(GADFmatrix)
	MTFimage.append(MTFmatrix)
	RGBimage.append(RGBmatrix)
    
    GASFimage = np.asarray(GASFimage)     #Convert the input to an array.Input should be the array_like.  
    GADFimage = np.asarray(GADFimage)
    MTFimage = np.asarray(MTFimage)
    RGBimage = np.asarray(RGBimage);print("images")

    ## draw large image and paa image
    #csvf=open("../data/changefile.csv").readlines()
    with open("../data/changefile.csv") as csvf:
	k=0
	reader=csv.reader(csvf)
        for row in reader:
	    GASFpath=os.path.join("../GASF",row[0]+".png")
	    GADFpath=os.path.join("../GADF",row[0]+".png")
	    MTFpath=os.path.join("../MTF",row[0]+".png")
	    RGBpath=os.path.join("../RGB",row[0]+".png")
            plt.imsave(GASFpath,GASFimage[k],format="png")#imsave(data)required data is 2-D array
	    plt.imsave(GADFpath,GADFimage[k],format="png")
	    plt.imsave(MTFpath,MTFimage[k],format="png")
	    plt.imsave(RGBpath,RGBimage[k],format="png")
	    print("save "+row[0]+" sucess!!!!")
	    k+=1
    print("success")

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import  sys

#PAA function
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

#Rescale data into [0,1]
def rescale(serie):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    return [(each-minval)/gap for each in serie]

#Rescale data into [-1,1]    
def rescaleminus(serie):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    return [(each-minval)/gap*2-1 for each in serie]

#################################
###Define the parameters here####
#################################
if __name__=="__main__":
    datafiles = ['../data/changefile.csv'] # Data fine name
    trains = [28] # Number of training instances (because we assume training and test data are mixed in one file)
    size = [64]  # PAA size
    GAF_type = 'GADF' # GAF type: GASF, GADF
    save_PAA = True # Save the GAF with or without dimension reduction by PAA: True, False
    rescale_type = 'Zero' # Rescale the data into [0,1] or [-1,1]: Zero, Minusone 

    for datafile, train in zip(datafiles,trains):
        fn = datafile
        for s in size:  
            print 'read file', datafile, 'size',s, 'GAF type', GAF_type
            raw = open(fn).readlines()
            raw = [map(float, each.strip().split(',')) for each in raw]
            length = len(raw[0])-1
            
	    image = []
            paaimage = []
            patchimage = []
            matmatrix = []
            fullmatrix = []
            for each in raw:
                if rescale_type == 'Zero':
                    std_data = rescale(each[1:])
                elif rescale_type == 'Minusone':
                    std_data = rescaleminus(each[1:])
                else:
                    sys.exit('Unknown rescaling type!')
                paalistcos = paa(std_data,s,None) 
            
                ################raw###################                
                datacos = np.array(std_data)    #create an array.
		print(type(std_data),std_data.ndim,"!!!!!!!!!!")
		print(type(datacos),datacos.ndim,"cos------------")
                datasin = np.sqrt(1-np.array(std_data)**2)
		print(type(datasin),datasin.ndim,"sin")

                paalistcos = np.array(paalistcos)
                paalistsin = np.sqrt(1-paalistcos**2)
            
                datacos = np.matrix(datacos)    #Returens a matrix from an array-like object,or form a string of data. A matrix is a specialized 2-D array that retains its 2-D nature through operations.Input should be the array_like of string.
                datasin = np.matrix(datasin)            
            
                paalistcos = np.matrix(paalistcos)
                paalistsin = np.matrix(paalistsin)            
                if GAF_type == 'GASF':
                    paamatrix = paalistcos.T*paalistcos-paalistsin.T*paalistsin
                    matrix = np.array(datacos.T*datacos-datasin.T*datasin)
                elif GAF_type == 'GADF':
                    paamatrix = paalistsin.T*paalistcos-paalistcos.T*paalistsin
                    matrix = np.array(datasin.T*datacos - datacos.T*datasin)
                else:
                    sys.exit('Unknown GAF type!')
                paamatrix = np.array(paamatrix)
                image.append(matrix)
                paaimage.append(np.array(paamatrix))
                matmatrix.append(paamatrix.flatten())
                fullmatrix.append(matrix.flatten())        #flatten(),return a copy of the array collapsed into one dimension.Return the array.
    
            image = np.asarray(image)        #Convert the input to an array.Input should be the array_like.         
            paaimage = np.asarray(paaimage)
            patchimage = np.asarray(patchimage)
            matmatrix = np.asarray(matmatrix)
            fullmatrix = np.asarray(fullmatrix)
        
            if save_PAA == False:        
                finalmatrix = matmatrix
            else:
                finalmatrix = fullmatrix
    print(len(std_data),type(std_data))  #----------------------------------------------------
    print(len(raw),type(raw),type(raw[0]))
    
    ## draw large image and paa image
    k=0
    plt.figure();
    plt.title(GAF_type + 'with PAA');
    plt.imshow(paaimage[k]);             #Display an image on the axws. Input should be the array_like.
    #plt.savefig("paa32.png")
    plt.show()

    '''
    k=0
    plt.figure()
    plt.suptitle("show result of different parameters.")

    ax1=plt.subplot(121)
    plt.imshow(image[k])
    #divider=make_axes_locatable(ax1)
    #cax=divider.append_axes("left",size="5%",pad=0.2)
    #plt.colorbar(cax=cax)

    ax2=plt.subplot(122)
    plt.imshow(paaimage[k])
    divider=make_axes_locatable(ax2)
    cax=divider.append_axes("right",size="5%",pad=0.2)
    plt.colorbar(cax=cax)
    
    plt.savefig('all.png')
    plt.show()
    
    print(type(image[k]),type(paaimage[k]),type(matmatrix[k]),type(fullmatrix[k])) 
    print(image[k].ndim,paaimage[k].ndim,matmatrix[k].ndim,fullmatrix[k].ndim)
    '''







    print("success")

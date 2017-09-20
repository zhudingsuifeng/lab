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

#Generate quantile bins
def QMeq(series, Q):
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    MSM = np.zeros([Q,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0, len(label)-1):
        MSM[label[i]][label[i+1]] += 1
    for i in xrange(Q):
        if sum(MSM[i][:]) == 0:
            continue
        MSM[i][:] = MSM[i][:]/sum(MSM[i][:])
    return np.array(MSM), label, q.levels

#Generate quantile bins when equal values exist in the array (slower than QMeq)
def QVeq(series, Q):
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    qv = np.zeros([1,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0,len(label)):
        qv[0][label[i]] += 1.0        
    return np.array(qv[0][:]/sum(qv[0][:])), label

#Generate Markov Matrix given a spesicif number of quantile bins
def paaMarkovMatrix(paalist,level):
    paaindex = []
    for each in paalist:    
        for k in range(len(level)):
            lower = float(level[k][1:-1].split(',')[0])
            upper = float(level[k][1:-1].split(',')[-1])
            if each >=lower and each <= upper:
                paaindex.append(k)
    return paaindex

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
      
            label = []
            image = []
            paaimage = []
            patchimage = []
            matmatrix = []
            fullmatrix = []
            for each in raw:
                label.append(each[0])
                if rescale_type == 'Zero':
                    std_data = rescale(each[1:])
                elif rescale_type == 'Minusone':
                    std_data = rescaleminus(each[1:])
                else:
                    sys.exit('Unknown rescaling type!')
                paalistcos = paa(std_data,s,None) 
            
                ################raw###################                
                datacos = np.array(std_data)
                datasin = np.sqrt(1-np.array(std_data)**2)

                paalistcos = np.array(paalistcos)
                paalistsin = np.sqrt(1-paalistcos**2)
            
                datacos = np.matrix(datacos)
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
                fullmatrix.append(matrix.flatten())
    
            label = np.asarray(label)
            image = np.asarray(image)
            paaimage = np.asarray(paaimage)
            patchimage = np.asarray(patchimage)
            matmatrix = np.asarray(matmatrix)
            fullmatrix = np.asarray(fullmatrix)
        
            if save_PAA == False:        
                finalmatrix = matmatrix
            else:
                finalmatrix = fullmatrix

    # polar coordinates
    k=0;
    r = np.array(range(1,length+1));
    r=r/100.0;
    theta = np.array(rescale(raw[k][1:]))*2*np.pi;

    #plt.figure();
    #plt.plot(theta, r, color='r', linewidth=3);
    #plt.show()

    ## draw large image and paa image
    plt.figure();
    plt.title(GAF_type + 'with PAA');
    ax=plt.subplot(111)
    #print(type(ax),ax)
    #bx=plt.plot(paaimage[k])
    #print(type(bx),bx)
    plt.imshow(paaimage[k]);
    divider=make_axes_locatable(ax)
    cax=divider.append_axes("right",size="5%",pad=0.2)
    plt.colorbar(cax=cax)
    plt.show()
    print("success")

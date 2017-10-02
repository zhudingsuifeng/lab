#!/usr/bin/env python
#coding=utf-8
"""
Create on Sun Oct 1 21:21:55 2017
@author: fly
A library of functions for compare image similarity and build complex networks.
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import warnings as wn
import cv2
import skimage.measure as sm
import networkx as nx
import pandas as pd

#Encoding the change price to compare the trend of the ups and downs.
def encoding(path):   #input is a file ,file format ['stockcode',2.0,-3.0,...]
    encoding=[]
    with open(path) as csvf:
	reader=csv.reader(csvf)
	i=0
	for row in reader:
	    encoding.append([row[0]])
	    for col in range(1,len(row)):
		if(float(row[col])==0):
		    encoding[i].append(0)
		elif(float(row[col])>0):
		    encoding[i].append(1)
		else:
		    encoding[i].append(-1)
	    i+=1
    csvf.close()
    return encoding        #return encoding 2-D array ,format as to input file.

#compute the hamming distance.
def hamming(serieA,serieB):#input serie is int type.
    binserie=bin(serieA^serieB)  #bin the serie.
    distance=binserie.count('1')  #distance of two series.
    return 1-distance/float((len(binserie)-2))    #return similarity of two series.'-2' because of bin has ob ahead.

def pHash(imagedir):
    print("success")

def sift(imagedir):
    print("success")

if __name__=="__main__":
    wn.filterwarnings("ignore")



    ''' 
    temp='../temp'
    nrmse=[]     #nrmse
    psnr=[]      #psnr
    ssim=[]      #ssim
    ssim,psnr,nrmse=compare('../GADF')
    plt.figure()
    plt.title('GADF')
    plt.plot(ssim,color="red",linewidth=1,label="ssim")
    plt.plot(psnr,color="blue",linewidth=1,label="psnr")
    plt.plot(nrmse,color="yellow",linewidth=1,label="nrmse")
    plt.legend(loc='upper left')
    plt.savefig('GADF.png',dpi=1000)
    plt.show()
    #ssim,psnr,nrmse=compare('../GADF')
    #ssim,psnr,nrmse=compare('../MTF')
    '''
    print("success")

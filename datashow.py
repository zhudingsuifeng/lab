#!/usr/bin/env python
#coding=utf-8
"""
Create on Thu Aug 24 21:01:35 2017
Modified on Wed Sep 27 11:22:00 2017
@author: fly
Compare picture various similarity and seve to file.
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

#compare image similarity
def compare(imagedir):
    ssimarray=[]
    psnrarray=[]
    nrmsearray=[]
    filelist=os.listdir(imagedir)      #list files of the dir
    for imdir in range(0,len(filelist)):
	imagepath=os.path.join(imagedir,filelist[imdir]) #combine the file path
	original=cv2.imread(imagepath)    #load the images
	#original=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY) #convert the image to grayscale
	for secimdir in range(0,len(filelist)):
	    secimagepath=os.path.join(imagedir,filelist[secimdir])
	    secimage=cv2.imread(secimagepath)
	    #secimage=cv2.cvtColor(secimage,cv2.COLOR_BGR2GRAY)
	    ssimvalue=sm.compare_ssim(original,secimage,multichannel=True)   #ssim
	    psnrvalue=sm.compare_psnr(original,secimage)                     #psnr  
	    nrmsevalue=sm.compare_nrmse(original,secimage)                    #nrmse   
	    print(filelist[secimdir]+" complete!")
	    ssimarray.append(ssimvalue)
	    psnrarray.append(psnrvalue)
	    nrmsearray.append(nrmsevalue)
	print("one success!")
	break
    return ssimarray,psnrarray,nrmsearray

if __name__=="__main__":
    wn.filterwarnings("ignore")

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

    print("success")

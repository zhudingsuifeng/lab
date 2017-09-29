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
def getSSIM(SSIM):
    path=os.path.join('../data',SSIM) #combine the file path
    ssim=np.load(path)    #load the npy file
    return ssim[0][1:]

if __name__=="__main__":
    wn.filterwarnings("ignore")

    temp='../temp'
    GASF=[]    
    GADF=[]     
    MTF=[]     
    GASF=getSSIM('GASFssim.npy')
    GADF=getSSIM('GADFssim.npy')
    MTF=getSSIM('MTFssim.npy')
    plt.figure()
    plt.title('SSIM')
    plt.plot(GASF,color="red",linewidth=1,label="GASF")
    plt.plot(GADF,color="blue",linewidth=1,label="GADF")
    plt.plot(MTF,color="yellow",linewidth=1,label="MTF")
    plt.legend(loc='upper left')
    plt.savefig('SSIM.png',dpi=1000)
    plt.show()
    #ssim,psnr,nrmse=compare('../GADF')
    #ssim,psnr,nrmse=compare('../MTF')

    print("success")

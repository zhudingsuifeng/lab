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

#the 'Mean Squared Error' between the two images is the sum of the squared difference between the two images
#Note:the two images must have the same dimension
def mse(imageA,imageB):
    err=np.sum((imageA.astype("float")-imageB.astype("float"))**2)
    err/=float(imageA.shape[0]*imageA.shape[1])
    #return the MSE,the lower the error ,the more "similar" the two images are
    return err

#save compare result to file
def save(ssim,savedir,title):
    #for inputarray,category in (ssim,'ssim'):
    path=os.path.join(savedir,title+"ssim.csv")         #combine path
    np.savetxt(path,ssim,delimiter=',')     #save to npy file,converted data to string by default.
    print("Save "+title+" success!")

#compare image similarity
def compare(imagedir):
    ssimarray=[]
    #nrmsearray=[]
    filelist=os.listdir(imagedir)      #list files of the dir
    for name in range(0,len(filelist)):
	(shortname,extension)=os.path.splitext(filelist[name])
	ssimarray.append([])
	#nrmsearray.append([])
    for imdir in range(0,len(filelist)):
	imagepath=os.path.join(imagedir,filelist[imdir]) #combine the file path
	original=cv2.imread(imagepath)    #load the images
	#original=cv2.cvtColor(original,cv2.COLOR_BGR2GRAY) #convert the image to grayscale
	for secimdir in range(0,len(filelist)):
	    secimagepath=os.path.join(imagedir,filelist[secimdir])
	    secimage=cv2.imread(secimagepath)
	    #secimage=cv2.cvtColor(secimage,cv2.COLOR_BGR2GRAY)
	    ssimvalue=sm.compare_ssim(original,secimage,multichannel=True)   #ssim
	    #psnrvalue=sm.compare_psnr(original,secimage)                     #psnr  
	    #nrmsevalue=sm.compare_nrmse(original,secimage)                    #nrmse   
	    print(filelist[secimdir]+" complete!ssim value is "+str(ssimvalue))
	    ssimarray[imdir].append(ssimvalue)
	    #nrmsearray[imdir].append(nrmsevalue)
	print("one success!")
	#break
    return ssimarray#,nrmsearray

if __name__=="__main__":
    wn.filterwarnings("ignore")

    #savedir='/home/fly/mygit/data'
    savedir='/home/fly/mygit/data/similarity'
    #nrmse=[]     #nrmse
    ssim=[]      #ssim
    ssim=compare('/home/fly/mygit/images/mixing/chcltr/gadf')
    save(ssim,savedir,"gadf")
    ssim=compare('/home/fly/mygit/images/mixing/chcltr/gasf')
    save(ssim,savedir,"gasf")

    print("success")

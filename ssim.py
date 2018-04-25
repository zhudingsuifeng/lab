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

#save compare result to file
def save(ssim,savedir,title):
    #for inputarray,category in (ssim,'ssim'):
    path=os.path.join(savedir,title+"ssim.csv")         #combine path
    np.savetxt(path,ssim,delimiter=',')     #save to npy file,converted data to string by default.
    print("Save "+title+" success!")

#compare image similarity
def compare(imagedir):
    ssimarray=[]
    filelist=os.listdir(imagedir)      #list files of the dir
    for name in range(0,len(filelist)):
	#(shortname,extension)=os.path.splitext(filelist[name])
	ssimarray.append([])
    for imdir in range(0,len(filelist)):
	imagepath=os.path.join(imagedir,filelist[imdir]) #combine the file path
	original=cv2.imread(imagepath)    #load the images
	for secimdir in range(0,len(filelist)):
	    secimagepath=os.path.join(imagedir,filelist[secimdir])
	    secimage=cv2.imread(secimagepath)
	    ssimvalue=sm.compare_ssim(original,secimage,multichannel=True)   #ssim
	    #print(filelist[secimdir]+" complete!ssim value is "+str(ssimvalue))
	    ssimarray[imdir].append(ssimvalue)
	print("one compare success!")
	#break
    return ssimarray

#get stock code
def get_stock_code(stockdir):
    filelist=os.listdir(stockdir)
    stockcode=[]
    for i in range(0,len(filelist)):
	(sn,ext)=os.path.splitext(filelist[i])
	stockcode.append(sn)
    with open('/home/fly/hs/interdata/stockcode.csv','w') as csvf:
	writer=csv.writer(csvf)
	writer.writerow(stockcode)
	csvf.close()

if __name__=="__main__":
    wn.filterwarnings("ignore")
    savedir='/home/fly/hs/interdata'
    ssim=[]      #ssim
    cssim=compare('/home/fly/hs/cgasf')
    save(cssim,savedir,"cgasf")
    lssim=compare('/home/fly/hs/lgasf')
    save(lssim,savedir,"lgasf")
    #get_stock_code('/home/fly/hs/data')

    print("success")

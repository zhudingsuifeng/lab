#!/usr/bin/env python
#coding=utf-8
"""
Create on Wed Dec 20 15:33:12 2017
@author: fly
Compare picture structural similarity and seve to file.
"""
import os
import csv
import numpy as np
import cv2
import skimage.measure as sm
import pandas as pd

#save compare result to file
def save(ssim,savedir,title):
    path=os.path.join(savedir,title+"ssim.csv")         #combine path
    np.savetxt(path,ssim,delimiter=',')     #save to npy file,converted data to string by default.
    print("Save "+title+" success!")

#compare image similarity
def compare(imagedir):
    ssimarray=[]
    filelist=os.listdir(imagedir)      #list files of the dir
    for name in range(0,len(filelist)):
	(shortname,extension)=os.path.splitext(filelist[name])
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
	print("calculate "+filelist[imdir]+" success!")
    return ssimarray

if __name__=="__main__":
    savedir='/home/fly/mygit/filter/result'
    ssim=[]      #ssim
    ssim=compare('/home/fly/mygit/filter/images')
    save(ssim,savedir,"plate")

    print("success")

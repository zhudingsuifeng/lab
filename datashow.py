#!/usr/bin/env python
#coding=utf-8
"""
Create on Thu Aug 24 21:01:35 2017
Modified on Mon Dec 4 09:43:26 2017
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
def getdata(datapath):
    path=os.path.join('/home/fly/mygit/data/similarity',datapath) #combine the file path
    data=np.loadtxt(path,delimiter=",")    #load the csv file
    return data[0]
    #floatdata=[]
    #return map(float,data[0][1:])     #map(func,data)   #data is a array-like series

if __name__=="__main__":
    wn.filterwarnings("ignore")
    filelist=os.listdir('/home/fly/mygit/data/similarity')
    label=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for name in range(0,len(filelist)):
	bardata=[]
	data=getdata(filelist[name])
	order=np.sort(data)
	for l in range(0,len(label)-1):
	    numi=0
	    for i in order:
		if i>label[l] and i<=label[l+1]:
		    numi+=1
	    bardata.append(numi)
	(shortname,extension)=os.path.splitext(filelist[name])
	plt.figure()
	plt.title(shortname)
	plt.bar(np.arange(10),bardata,facecolor='g')    #draw the bar image
	plt.xticks(np.arange(10),('0.0~0.1','0.1~0.2','0.2~0.3','0.3~0.4','0.4~0.5','0.5~0.6','0.6~0.7','0.7~0.8','0.8~0.9','0.9~1.0'),rotation=30)
	#plt.xticks()  #set the xticks ,rotation=90,rotate 30 degrees
	path=os.path.join('/home/fly/mygit/images/similarity',shortname+'.png')
	plt.savefig(path,dpi=100)
	#plt.show()
	plt.close()
	#break
	
    print("success")

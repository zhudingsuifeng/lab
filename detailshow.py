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

if __name__=="__main__":
    wn.filterwarnings("ignore")
    filelist=os.listdir('/home/fly/mygit/data/similarity')
    for name in range(0,len(filelist)):
	bardata=[]
	Tks=[]
	for i in np.arange(0,1,0.1):
	    Tks.append(str(i))
	data=getdata(filelist[name])
	order=np.sort(data)
	for l in np.arange(0,1,0.01):
	    numi=0
	    for i in order:
		if i>l and i<=(l+0.01):
		    numi+=1
	    bardata.append(numi)
	(shortname,extension)=os.path.splitext(filelist[name])
	plt.figure()
	plt.title(shortname)
	plt.bar(np.arange(100),bardata,facecolor='g')    #draw the bar image
	plt.xticks(np.arange(0,100,10),(Tks),rotation=0)
	#plt.xticks()  #set the xticks ,rotation=90,rotate 30 degrees
	path=os.path.join('/home/fly/mygit/images/similarity',shortname+'.png')
	plt.savefig(path,dpi=100)
	#plt.show()
	plt.close()
	print(shortname+" success")
	#break
	
    print("success")

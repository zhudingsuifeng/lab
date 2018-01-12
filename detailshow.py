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
import cv2
import skimage.measure as sm
import networkx as nx
import pandas as pd

#compare image similarity
def getdata(datapath):
    #path=os.path.join('/home/fly/mygit/data/similarity',datapath) #combine the file path
    data=np.loadtxt(datapath,delimiter=",")    #load the csv file
    return data

if __name__=="__main__":
    fpath='/home/fly/mygit/data/similarity/gasfssim.csv'
    Tks=[]
    data=getdata(fpath)
    for i in np.arange(0,1,0.1):
	Tks.append(str(i))
    for row in range(0,len(data)):
	bardata=[]
	order=np.sort(data[row])
	for l in np.arange(0,1,0.01):
	    numi=0
	    for i in order:
		if i>l and i<=(l+0.01):
		    numi+=1
	    bardata.append(numi)
	plt.figure()
	#plt.title(shortname)
	plt.bar(np.arange(100),bardata,facecolor='g')    #draw the bar image
	plt.xticks(np.arange(0,100,10),(Tks),rotation=0)  #ticks 
	plt.xlabel("Similarity")
	plt.ylabel("Number of shares")
	#plt.xticks()  #set the xticks ,rotation=90,rotate 30 degrees
	path=os.path.join('/home/fly/mygit/images/similarity','gasfssim'+str(row)+'.png')
	plt.savefig(path,dpi=100)
	#plt.show()
	plt.close()
	print("gasfssim "+str(row)+" success")
	#break
	
    print("success")

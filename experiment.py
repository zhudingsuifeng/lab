#!/usr/bin/env python
#coding=utf-8
"""
Create on Mon Dec 11 09:35:01 2017
@author: fly
Find the high similarity images.
opencv3.3.0
numpy1.13.1
pandas0.20.3
PIL4.2.1
skimage0.13.0
matplotlib(2.0.2)
"""
import os
import csv
import sys
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import warnings as wn
import cv2
import skimage.measure as sm
import networkx as nx
import pandas as pd
from PIL import Image
from compiler.ast import flatten

#draw the match points in two images
def matchimages(imga,imgb,sift,threshold):
    #detect key points and compute descriptor by sift
    kpa,desa=sift.detectAndCompute(imga,None)
    kpb,desb=sift.detectAndCompute(imgb,None)

    #match key points by flann
    FLANN_INDEX_KDTREE=0
    index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    search_params=dict(checks=50)
    flann=cv2.FlannBasedMatcher(index_params,search_params)
    if len(kpa)>=2 and len(kpb)>=2:
	matches=flann.knnMatch(desa,desb,k=2)   #knnMatch(desa,desb,k=2) desa,desa are the feature point descriptor sequence,k=2,select 2 best matches.

	goodmatch=[]
	#ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
	    if m.distance<threshold*n.distance:    #adjustable parameters
	    	goodmatch.append(i)
    	return len(goodmatch)
    else:
	return 0

if __name__=="__main__":
    wn.filterwarnings("ignore")
    print("Test case.")
    #imgdir='/home/fly/mygit/images/mixing/chcltr/gadf'
    imgdir='/home/fly/mygit/images/trade/GASF'
    matchdir='/home/fly/mygit/data/matches'
    savedir='/home/fly/mygit/images/temp/trade'
    china='601988.png'
    haier='600690.png'
    stock=0
    filelist=os.listdir(imgdir)          #list the file name in the dir
    imagelist=[]
    sift=cv2.xfeatures2d.SIFT_create()   #create a sift object
    surf=cv2.xfeatures2d.SURF_create()   #create a surf object
    #cv2.imread() all images
    for i in range(0,len(filelist)):
	imgpath=os.path.join(imgdir,filelist[i])
	img=cv2.imread(imgpath,0)
	imagelist.append(img)
    print("load images success")

    for i in range(0,len(filelist)):
	if haier==filelist[i]:      #stock name---------------------------------------------------
	    stock=i
	    print(stock)    

    #draw key points and matches
    for hold in np.arange(0.1,1,0.1):
	matchstock=[]
	for j in range(0,len(filelist)):
	    #(namea,ext)=os.path.splitext(filelist[i])
	    (nameb,ext)=os.path.splitext(filelist[j])
	    suitability=matchimages(imagelist[stock],imagelist[j],sift,hold)#threshold-----------------
	    #Fourth parameter is threshold which could adjust.
	    matchstock.append([nameb,suitability])
	#print(" success++")
	
	matchstockinf=[]
	stockinf=ts.get_stock_basics()   #get stock data from tushare
	#stockcode=stockinf.index         #get stock code
	with open("/home/fly/mygit/data/stock/basic.csv","r") as csvf:
	    reader=csv.reader(csvf)
	    for line in reader:
		#sort the stock with suitability.
		#for code,suit in sorted(matchstock,key=lambda x : x[1],reverse=True):
		for i in range(0,len(matchstock)):
		    #print('%s %d' % (name,suit))
		    if matchstock[i][0]==line[0]:
			matchstockinf.append([line[0],line[1],matchstock[i][1],line[2],line[3]])
	    	#break
    	print("sort success")
    	#print(matchstockinf)
     
    	#save data as csv file.
    	(shortname,ext)=os.path.splitext(haier)       #save path--------------------------------------
    	path=os.path.join(savedir,shortname+'hold'+str(hold)+'.csv')
    	csvf=open(path,"w")   #open csv file
    	writer=csv.writer(csvf)
    	stockinf=sorted(matchstockinf,key=lambda x : x[2],reverse=True)    #sort the data
    	for i in range(0,20):   #20 high suitability stock
	    writer.writerow(stockinf[i])     #write data row to csv file
    	csvf.close()    #close csv file 
    	print("save "+str(hold)+" file success")
    
    print("success")

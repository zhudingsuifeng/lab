#!/usr/bin/env python
#coding=utf-8
"""
Create on Sun Dec 10 21:07:46 2017
@author: fly
Draw key points in images and match of images.
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
import matplotlib.pyplot as plt
import warnings as wn
import cv2
import skimage.measure as sm
import networkx as nx
import pandas as pd
from PIL import Image
from compiler.ast import flatten

#draw the key points in images
def keypointimage(img,sift,keypath):
    gray=img.copy()
    kp,des=sift.detectAndCompute(img,None)
    #kp,des=sift.detectAndCompute(gray,None)
    fig,ax=plt.subplots(1,1,figsize=(10,10),dpi=10)    #set fig size
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0,hspace=0)   #remove the margins
    ax.axis('off')    #delete the scale 
    #outimg=cv2.drawKeypoints(img,kp,gray)    #draw the key points in picture
    outimg=cv2.drawKeypoints(img,kp,gray)
    plt.imshow(outimg)
    plt.savefig(keypath)
    plt.close()

#draw the match points in two images
def matchimages(imga,imgb,sift,matchpath):
    #detect key points and compute descriptor by sift
    kpa,desa=sift.detectAndCompute(imga,None)
    kpb,desb=sift.detectAndCompute(imgb,None)

    #match key points by flann
    FLANN_INDEX_KDTREE=0
    index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    search_params=dict(checks=50)
    flann=cv2.FlannBasedMatcher(index_params,search_params)
    matches=flann.knnMatch(desa,desb,k=2)   #knnMatch(desa,desb,k=2) desa,desa are the feature point descriptor sequence,k=2,select 2 best matches.
    #When k<2,error:(-215)(globalDescIdx>=0)&&(globalDescIdx<size()) in function getLocalIdx.
    #When you apply knn match,make sure that number of featrues(keypoints) in both images is greater than or equal to number of nearest neighbors in knn match.

    #set match params
    matchesMask=[[0,0] for i in xrange(len(matches))]
    #ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
	if m.distance<0.6*n.distance:    #adjustable parameters
	    matchesMask[i]=[1,0]
    draw_params=dict(matchColor=(0,255,0),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=0)
    matchimg=cv2.drawMatchesKnn(imga,kpa,imgb,kpb,matches,None,**draw_params)

    #draw matches key points in the images
    fig,ax=plt.subplots(1,1,figsize=(20,10),dpi=10)
    plt.subplots_adjust(left=0,right=1,top=1,bottom=0,wspace=0,hspace=0)
    ax.axis('off')
    plt.imshow(matchimg)
    plt.savefig(matchpath)
    plt.close()

if __name__=="__main__":
    wn.filterwarnings("ignore")
    print("Test case.")
    imgdir='/home/fly/mygit/images/mixing/chcltr/gadf'
    keydir='/home/fly/mygit/images/keypoints/gadf'
    matchdir='/home/fly/mygit/images/matches/gadf'
    filelist=os.listdir(imgdir)          #list the file name in the dir
    imagelist=[]
    sift=cv2.xfeatures2d.SIFT_create()   #create a sift object
    surf=cv2.xfeatures2d.SURF_create()   #create a surf object
    #cv2.imread() all images
    for i in range(0,len(filelist)):
	imgpath=os.path.join(imgdir,filelist[i])
	img=cv2.imread(imgpath,0)
	keypath=os.path.join(keydir,filelist[i])
	keypointimage(img,sift,keypath)
	imagelist.append(img)
	print("load images success")

    #draw key points and matches
    for i in range(0,len(filelist)):
	#keypath=os.path.join(keydir,filelist[i])
	#keypointimage(img,sift,keypath)
	for j in range(0,len(filelist)):
	    (namea,ext)=os.path.splitext(filelist[i])
	    (nameb,ext)=os.path.splitext(filelist[j])
	    matchpath=os.path.join(matchdir,namea+"and"+nameb+".png")
	    matchimages(imagelist[i],imagelist[j],sift,matchpath)
	    break
	print(str(i)+" success++")
	break
    print("success")

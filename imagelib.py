#!/usr/bin/env python
#coding=utf-8
"""
Create on Sun Oct 1 21:21:55 2017
@author: fly
A library of functions for compare image similarity and build complex networks.very time consuming.
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

#compute the hamming distance.
def hamming(serieA,serieB):#input serie is int type.
    #print(serieA)
    serieA=int(serieA,2)
    serieB=int(serieB,2)
    binserie=bin(serieA^serieB)  #bin the serie.
    distance=binserie.count('1')  #distance of two series.
    return 1-distance/float((len(binserie)-2))    #return similarity of two series.'-2' because of bin has ob ahead.

#mean hash,get image hash string
def aHash(path):
    img=Image.open(path)
    img=img.resize((8,8),Image.ANTIALIAS).convert('L')#ANTIALIAS anti-aliasing,'L' convert to grayscale.
    avg=sum(list(img.getdata()))/64.0
    return ''.join(map(lambda i: '0' if i<avg else '1',img.getdata()))
    #return ''.join(map(lambda x:'%x' % int(strhash[x:x+4],2),range(0,64,4)))#convert binary to hexadecimal.

#perceptual hash algorithm,get image pHash value
def pHash(path):
    #reconvert image to 32*32 gray image
    img=cv2.imread(path,0)
    img=cv2.resize(img,(32,32),interpolation=cv2.INTER_CUBIC)
    #create 2-d list
    h,w=img.shape[:2]
    vis0=np.zeros((h,w),np.float32)
    vis0[:h,:w]=img
    #2-d Dct transform
    vis1=cv2.dct(cv2.dct(vis0))
    #resize image
    vis1=vis1[0:8,0:8]#get upper left corner 8*8 value,not resize(8,8)
    img_list=flatten(vis1.tolist())
    avg=sum(img_list)*1./len(img_list)
    return ''.join(['0' if i<avg else '1' for i in img_list])

#get image dHash value
def dHash(path):
    img=cv2.imread(path,0)
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    img=np.int8(img)#int type,out of range is negative number.
    img_diff=img[:,:-1]-img[:,1:]   #get value,-1 is the last, but list get value is [).
    img_list=flatten(img_diff.tolist())
    avg=sum(img_list)*1./len(img_list)#get the average
    return ''.join(['0' if i<avg else '1' for i in img_list])

#calculated the histogram similarity.
def histsim(hista,histb):
    degree=0
    for i in range(len(hista)):
	if hista[i][0]!=histb[i][0]:
	    degree+=(1-abs(hista[i][0]-histb[i][0])/max(hista[i][0],histb[i][0]))
	else:
	    degree+=1
    degree=degree/len(hista)
    return degree

#get color histogram from RGB images.
def histRGB(patha,pathb):
    imga=cv2.imread(patha)
    imgb=cv2.imread(pathb)
    imga=cv2.split(imga)
    imgb=cv2.split(imgb)
    degree=0
    for ima,imb in zip(imga,imgb):
	hista=cv2.calcHist([ima],[0],None,[256],[0,256])
	histb=cv2.calcHist([imb],[0],None,[256],[0,256])
	degree+=histsim(hista,histb)
    degree=degree/3.
    return degree

#FAST+SIFT
def FASTSIFT(patha,pathb):
    imga=cv2.imread(patha,0)
    imgb=cv2.imread(pathb,0)      #RGB->GRAY ,Y<-0.299R+0.587G+0.114B
    #Initiate FAST object with default values
    fast=cv2.FastFeatureDetector_create()
    #fast=cv2.FastFeatureDetector()#AttributeError:'module' object has no attribute 'FastFeatureDetector'
    kpa=fast.detect(imga,None)
    kpb=fast.detect(imgb,None)
    sift=cv2.xfeatures2d.SIFT_create()
    kpa,desa=sift.compute(imga,kpa)
    kpb,desb=sift.compute(imgb,kpb)
    good=flannKnnMatcher(desa,desb,'spot')
    return good/float(max(len(kpa),len(kpb)))

#HARRIS+SIFT (goodFeaturesToTrack)
def HARRISSIFT(patha,pathb):
    imga=cv2.imread(patha,0)      #RGB->GRAY ,Y<-0.299R+0.587G+0.114B
    imgb=cv2.imread(pathb,0)
    #kpa=cv2.cornerHarris(imga,2,3,0.04)#cornerHarris()return the Harris response as the narray ,size as the img.
    gftt=cv2.GFTTDetector_create()#Initiate the GFTTDetector,use the Harris principle,for unified interface.
    kpa=gftt.detect(imga,None)
    kpb=gftt.detect(imgb,None)
    sift=cv2.xfeatures2d.SIFT_create()
    kpa,desa=sift.compute(imga,kpa)
    kpb,desb=sift.compute(imgb,kpb)
    good=flannKnnMatcher(desa,desb,'spot')
    return good/float(max(len(kpa),len(kpb)))

#SURF+SIFT
def SURFSIFT(patha,pathb):
    imga=cv2.imread(patha,0)        #RGB->GRAY ,Y<-0.299R+0.587G+0.114B
    imgb=cv2.imread(pathb,0)
    #Initiate SURF detector.
    surf=cv2.xfeatures2d.SURF_create()#create the surf object.
    kpa=surf.detect(imga,None)
    kpb=surf.detect(imgb,None)
    sift=cv2.xfeatures2d.SIFT_create()#create the sift object.
    kpa,desa=sift.compute(imga,kpa)
    kpb,desb=sift.compute(imgb,kpb)
    good=flannKnnMatcher(desa,desb,'spot')
    return good/float(max(len(kpa),len(kpb)))

#MSER+SIFT
def MSERSIFT(patha,pathb):
    imga=cv2.imread(patha,0)    #RGB->GRAY ,Y<-0.299R+0.587G+0.114B
    imgb=cv2.imread(pathb,0)
    #Initiate MSER detector.
    mser=cv2.MSER_create()
    kpa=mser.detect(imga,None)
    kpb=mser.detect(imgb,None)
    sift=cv2.xfeatures2d.SIFT_create() #create the sift object.
    kpa,desa=sift.compute(imga,kpa)
    kpb,desb=sift.compute(imgb,kpb)
    good=flannKnnMatcher(desa,desb,'spot')
    return good/float(max(len(kpa),len(kpb)))

#STAR+SIFT
def STARSIFT(patha,pathb):
    imga=cv2.imread(patha,0)    #RGB->GRAY ,Y<-0.299R+0.587G+0.114B
    imgb=cv2.imread(pathb,0)
    #Initiate STAR detector.
    star=cv2.xfeatures2d.StarDetector_create()#create the star object.
    kpa=star.detect(imga,None)
    kpb=star.detect(imgb,None)
    if len(kpa)==0 or len(kpb)==0:
	print("don't detect feature points")
	return 0
    sift=cv2.xfeatures2d.SIFT_create() #create the sift object.
    kpa,desa=sift.compute(imga,kpa)
    kpb,desb=sift.compute(imgb,kpb)
    good=flannKnnMatcher(desa,desb,'spot')
    return good/float(max(len(kpa),len(kpb)))

#SIFT+SIFT
def SIFTSIFT(patha,pathb):
    imga=cv2.imread(patha,0)   #RGB->GRAY ,Y<-0.299R+0.587G+0.114B
    imgb=cv2.imread(pathb,0)
    sift=cv2.xfeatures2d.SIFT_create()#create the sift object,different to others.
    kpa,desa=sift.detectAndCompute(imga,None)
    kpb,desb=sift.detectAndCompute(imgb,None)
    good=flannKnnMatcher(desa,desb,'spot')
    return good/float(max(len(kpa),len(kpb)))

#ORB+ORB+BForFLANN
def ORBORB(patha,pathb):
    imga=cv2.imread(patha,0)
    imgb=cv2.imread(pathb,0)  #RGB->GRAY ,Y<-0.299R+0.587G+0.114B
    #Initiate ORB detector
    orb=cv2.ORB_create()
    #find the keypoints and descriptors with ORB
    kpa,desa=orb.detectAndCompute(imga,None)
    kpb,desb=orb.detectAndCompute(imgb,None)
    if len(kpa)==0 or len(kpb)==0:
	print("don't detect feature points")
	return 0
    good=flannKnnMatcher(desa,desb,'corner')
    #good=bfKnnMatcher(desa,desb)
    return good/float(max(len(kpa),len(kpb)))

#BF
def bfKnnMatcher(desa,desb):
    #BFMatcher with default params.
    bf=cv2.BFMatcher()
    matches=bf.knnMatch(desa,desb,k=2)
    #Apply ratio test
    good=[]
    for m,n in matches:
	if m.distance<0.96*n.distance:
	    good.append([m]) 
    return len(good)

#FLANN for corner likes orb or spot likes sift surf.
def flannKnnMatcher(desa,desb,Type):
    if Type=='corner':
	FLANN_INDEX_LSH=6
	index_params=dict(algorithm=FLANN_INDEX_LSH,table_number=6,key_size=12,multiprobe_level=1)
    elif Type=='spot':
	FLANN_INDEX_KDTREE=0
	index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
    else:
	sys.exit('Unkonwn type!')
    search_params=dict(checks=50)
    flann=cv2.FlannBasedMatcher(index_params,search_params)
    matches=flann.knnMatch(desa,desb,k=2)
    good=[]
    for m in matches:
	if len(m)==2:
	    if m[0].distance<0.96*m[1].distance:
		good.append([m[0]])
    return len(good)

#compare image similarity
def compare(imagedir):
    surfsift=[]
    siftsift=[]
    orborb=[]
    phashham=[]
    histsim=[]
    filelist=os.listdir(imagedir)
    for name in range(0,len(filelist)):
	(shortname,extension)=os.path.splitext(filelist[name])
	surfsift.append([shortname])
	siftsift.append([shortname])
	orborb.append([shortname])
	phashham.append([shortname])
	histsim.append([shortname])
    for imdir in range(0,len(filelist)):
	imagepath=os.path.join(imagedir,filelist[imdir])
	serieA=pHash(imagepath)
	for secimdir in range(0,len(filelist)):
	    secimagepath=os.path.join(imagedir,filelist[secimdir])
	    surfsiftvalue=SURFSIFT(imagepath,secimagepath)
	    siftsiftvalue=SIFTSIFT(imagepath,secimagepath)
	    orborbvalue=ORBORB(imagepath,secimagepath)
	    serieB=pHash(secimagepath)
	    phashvalue=hamming(serieA,serieB)
	
	    histvalue=histRGB(imagepath,secimagepath)
	    print(filelist[secimdir]+" complete!")

	    surfsift[imdir].append(surfsiftvalue)
	    siftsift[imdir].append(siftsiftvalue)
	    orborb[imdir].append(orborbvalue)
	    histsim[imdir].append(histvalue)
	    phashham[imdir].append(phashvalue)
	print("one success!")
	break
    return surfsift,siftsift,orborb,histsim,phashham

#save compare result to file 
def save(surfsift,siftsift,orborb,histsim,phashham,savedir,title):
    for value,cate in (surfsift,'surfsift'),(siftsift,'siftsift'),(orborb,'orborb'),(histsim,'histsim'),(phashham,'phashham'):
	path=os.path.join(savedir,title+cate+".npy")
	np.save(path,value)
    print("Save "+title+" success!")

if __name__=="__main__":
    wn.filterwarnings("ignore")
    print("Test case.")
    surfsift,siftsift,orborb,histsim,phashham=compare('/home/fly/mygit/images/change/RGB')
    save(surfsift,siftsift,orborb,histsim,phashham,'../data',"RGB")
    print("success")

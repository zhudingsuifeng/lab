#!/usr/bin/env python
#coding=utf-8

'''
Created on Wed Jun 6 21:18:52 2018
contrast the methods to highlight the features of our method.
@author:fly
'''

import os
import csv
import networkx as nx
import warnings as wn
import matplotlib.pyplot as plt
import scipy.stats as stats 
import pandas as pd
import numpy as np
import cv2
import skimage.measure as sm

def gasf(series):     #transform serie to GASF image
	datacos=np.array(series)
	datasin=np.sqrt(1-np.array(series)**2)
	matcos=np.matrix(datacos)
	matsin=np.matrix(datasin)
	matrix=np.array(matcos.T*matcos-matsin.T*matsin)
	return matrix

#mixing r,g,b to rgb image
def mgasf(r,g,b,length):
	rgb=np.zeros([length,length,3])
	for i in range(0,length):
		for k in range(0,length):
			rgb[i][k][0]=r[i][k]
			rgb[i][k][1]=g[i][k]
			rgb[i][k][2]=b[i][k]
	return rgb

#
def mssim(n1,n2):
	p1=os.path.join('/home/fly/mygit/images/result',n1)
	p2=os.path.join('/home/fly/mygit/images/result',n2)
	c1=cv2.imread(p1)
	c2=cv2.imread(p2)
	s=sm.compare_ssim(c1,c2,multichannel=True)
	return s
#
def iplot(s,l,name):
	plt.figure(figsize=(10,10),dpi=30)
	plt.plot(s,color='limegreen',label=l)
	plt.legend(loc=1)
	path=os.path.join('/home/fly/mygit/images/result',name)
	plt.savefig(path)
	plt.close()

def savef(image,name):
    savepath=os.path.join('/home/fly/mygit/images/result',name)
    plt.imsave(savepath,image,format="png")

if __name__=='__main__':
	s1=[i for i in np.arange(0,1,0.01)]
	iplot(s1,'s1','s1.png')
	s2=[i for i in np.arange(0.5,1,0.01)]+[i for i in np.arange(0,0.5,0.01)]
	iplot(s2,'s2','s2.png')
	s3=[(np.sin(2*np.pi*i)+1)/2 for i in np.arange(0,2,0.02)]
	iplot(s3,'s3','s3.png')
	logs=[s1,s2,s3]
	pearson=np.corrcoef(logs)
	print("s1 and s3 similarity is "+str(pearson[0][2]))
	print("s2 and s3 similarity is "+str(pearson[1][2]))
	p1=gasf(s1)
	savef(p1,'p1.png')
	p2=gasf(s2)
	savef(p2,'p2.png')
	p3=gasf(s3)
	savef(p3,'p3.png')
	print("p1,p3 similarity is "+str(mssim('p1.png','p3.png')))
	print("p2,p3 similarity is "+str(mssim('p2.png','p3.png')))
	t=np.zeros([100,100])
	p13=mgasf(p1,p3,t,100)
	savef(p13,'p13.png')
	p23=mgasf(p2,p3,t,100)
	savef(p23,'p23.png')
	print("p13,p23 similarity is "+str(mssim('p13.png','p23.png')))

	print("success")


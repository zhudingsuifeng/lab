#!/usr/bin/env python
#coding=utf-8
"""
Create on Sun Dec 3 15:33:50 2017
Modified on Wed Apr 18 09:15:12 2018
@author: fly
Compare features similarity and show in the picture.
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

#create a heat map with label
def heat_map(matrix,imgname,label):
    matrix=np.array(matrix)
    #fig=plt.figure(figsize=(10,10))
    #ax=fig.add_subplot(1,1,1,frameon=False)
    fig,ax=plt.subplots()
    ax.imshow(matrix)
    ax.set_title("Feature Select")
    #ax.set_xticks(np.arange(len(label)))
    #ax.set_yticks(np.arange(len(label)))
    #ax.set_xticklabels(label)
    #ax.set_yticklabels(label)
    fig.tight_layout()
    plt.savefig(imgname)
    plt.show()

if __name__=="__main__":
    pearson=np.loadtxt('/home/fly/hs/data/000001.csv',delimiter=",")
    label=[]
    label=['change','%chg','open','high','close','low','volume','turnover','ma5','ma10','ma20','v_ma5','v_ma10','v_ma20']
    heat_map(pearson,"/home/fly/hs/p1.png",label)
    '''
    s=[[0.148,0.149,0.986,0.995,1,0.992,0.659,0.661],
       [0.997,1,0.015,0.095,0.149,0.043,0.353,0.353],
       [0.338,0.353,0.607,0.689,0.659,0.6,1,1]]
    name=['change','close','volume']
    for i in range(0,len(s)):    
	plt.figure(figsize=(10,10),dpi=60)
	plt.title('Pearson correlation',{'fontsize':36})
	plt.bar(np.arange(0,2,1),s[i][:2],facecolor='r')    #draw the bar image
	plt.bar(np.arange(2,6,1),s[i][2:6],facecolor='g')   #draw the bar with different color
	plt.bar(np.arange(6,8,1),s[i][6:8],facecolor='b')
	plt.tick_params(labelsize=26)
	plt.xticks(np.arange(8),('change','%chg','open','high','close','low','volume','turnover'),rotation=30)
	#plt.xticks()  #set the xticks ,rotation=90,rotate 30 degrees
	path=os.path.join('/home/fly/mygit/images/schematic',name[i]+'.png')
	plt.savefig(path)
	plt.close()
	#plt.show()
	#break
    '''	
    print("success")

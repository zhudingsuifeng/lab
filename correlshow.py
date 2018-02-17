#!/usr/bin/env python
#coding=utf-8
"""
Create on Sun Dec 3 15:33:50 2017
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

if __name__=="__main__":
    wn.filterwarnings("ignore")
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
    	
    print("success")

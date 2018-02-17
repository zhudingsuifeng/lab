#!/usr/bin/env python
#coding=utf-8
"""
Create on Mon Oct 30 21:46:32 2017
Modify on Thu Nov 2 16:06:59 2017
@author:fly
calculate the pearson correlation coefficient of each feature.
"""
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings as wn

datadir='/home/fly/temp/data'
filelist=os.listdir(datadir)   #list the file of the directory
savedir='/home/fly/mygit/data/pearson'
#sc=StandardScaler()
#pca=PCA(n_components=14)      #create a PCA object.
for f in range(0,len(filelist)):
    temp=[]
    path=os.path.join(datadir,filelist[f])
    (shortname,extemsion)=os.path.splitext(filelist[f])
    savepath=os.path.join(savedir,shortname+'.csv')
    data=np.load(path)       # load .npy file 
    for i in range(0,len(data)):
	temp.append([])
	temp[i]=map(float,data[i])    #map(func,data)   #using func every data
    temp=np.array(temp)
    var=np.std(temp,axis=0)
    p=np.corrcoef(temp,rowvar=0)   #Return Pearson product-moment correlation coefficients.
    for j in range(0,len(p)):
	for i in range(0,len(p)):
	    p[j][i]=("%.3f" % p[j][i])   #("%.3f" % x)  #keep 3 . decimal places
    np.savetxt(savepath,p,delimiter=',')  #save matrix to csv file 
    #pcadata=pca.fit(temp)
    #break
print("success")

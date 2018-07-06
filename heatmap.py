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
import pandas as pd
import seaborn as sns   #python visual toolkit
import matplotlib.pyplot as plt
import warnings as wn

hs='/home/fly/hs/data'
filelist=os.listdir(hs)   #list the file of the directory
savedir='/home/fly/hs'
for f in range(0,len(filelist)):
	path=os.path.join(hs,filelist[f])
	(sh,ext)=os.path.splitext(filelist[f])
	savepath=os.path.join(savedir,sh+".png")
	data=pd.read_csv(path,usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14])   #read csv file by using pandas
	data=data[['open','high','close','low','ma5','ma10','ma20','p_change','price_change','volume','turnover','v_ma5','v_ma10','v_ma20']]   #change data order
	df=data.corr()         #pearson correlation coefficient matrix 
	fig,ax=plt.subplots(figsize=(9.5,7))  #adjust window ratio
	#ax.set_title('Pearson Correlation')
	#sns.heatmap(df,annot=True,square=True,vmax=1,cmap="Blues")
	sns.set(font_scale=1.5)  #set color bar font size
	ax.tick_params(labelsize=15) #set ticklabels font size
	ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
	ax=sns.heatmap(df,annot=False,square=True,vmax=1,cmap="Blues")
	plt.savefig(savepath)
	plt.show()
	break
print("success")

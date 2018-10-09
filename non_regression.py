#!/usr/bin/env python2
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

def excel_to_list(p,l):
    d=pd.read_excel(p,usecols=l)
    d=np.array(d)
    d=d.tolist()
    

d='/home/fly/works/result.xlsx'
img='/home/fly/works/tend.png'
data=pd.read_excel(d,usecols=[0,4])   #read csv file by using pandas
df=data.corr()         #pearson correlation coefficient matrix 
fig,ax=plt.subplots(figsize=(9.5,7))  #adjust window ratio
#ax.set_title('Pearson Correlation')
#sns.heatmap(df,annot=True,square=True,vmax=1,cmap="Blues")
sns.set(font_scale=1.5)  #set color bar font size
ax.tick_params(labelsize=15) #set ticklabels font size
ax.set_xticklabels(ax.get_xticklabels(),rotation=60)
ax=sns.heatmap(df,annot=False,square=True,vmax=1,cmap="Blues")
plt.savefig(img)
plt.show()
print("success")

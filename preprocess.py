#!/usr/bin/env python
#coding=utf-8
"""
Create on Tue Dec 19 22:09:01 2017
@author: fly
Filter stocks by sector.
numpy1.13.1
pandas0.20.3
"""
import os
import csv
import sys
import numpy as np
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

#get total number of rows
def get_num_rows(datadir,fname):
	path=os.path.join(datadir,fname)
	rnum=0
	with open(path,'r')as f:
		csvf=csv.reader(f)
		for i in csvf:
			rnum+=1
	return rnum

#get date from index 
def get_date(path):
	date=[]
	with open(path,'r')as f:
		csvf=csv.reader(f)
		for row in csvf:
			date.append(row[1])
		return date[1:]

#get data with date
def get_data(fname,date):
	path=os.path.join('/home/fly/cs/csi300',fname)
	with open(path,'r')as f:
		csvf=csv.reader(f)
		for row in csvf:
			if date==row[1]:
				return row[2:7]
		return [0,0,0,0,0]

#calculate cosine similarity of vectors in a matrix
def cos_sim(matx):
	temp=cosine_similarity(matx)
	return temp

#save data to csv file
def savef(path,datas):
	with open(path,'w')as f:
		writer=csv.writer(f)
		writer.writerows(datas)

#number of occurrences of elements in the statistics list, and is displayed as a histogram
def stat(lst):
	d=dict()
	lst=sorted(lst)
	for i in lst:
		d[i]=d.get(i,0)+1
	index=[]
	value=[]
	for i in d:
		index.append(i)
		value.append(d.get(i))
	fig,ax=plt.subplots()
	ax.set_xlabel('days')
	ax.set_ylabel('value')
	ax.set_title('days distribution statistics')
	ax.set_xticks(np.arange(len(index)))
	ax.set_xticklabels(map(str,index))
	ax.bar(np.arange(len(index)),value,width=0.3,color='green')
	#plt.grid(True,linestyle='--')
	fname='/home/fly/cs/statistics.png'
	#plt.show()
	plt.savefig(fname)
	plt.close()

if __name__=="__main__":
	print("Test case.")
	calendar=get_date('/home/fly/cs/399300.csv')
	for d in calendar:
		mtx=[]
		for each in os.listdir('/home/fly/cs/csi300'):
			temp=get_data(each,d)
			mtx.append(temp)
		cs=cos_sim(mtx)
		path=os.path.join('/home/fly/cs/cos300',d+'.csv')
		savef(path,cs)
		break
	#stat(lst)
	print("success")

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
import math
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
	path=os.path.join('/home/fly/cs/data',fname)
	with open(path,'r')as f:
		csvf=csv.reader(f)
		for row in csvf:
			if date==row[1]:
				return row[2:6]
		return [0,0,0,0]

#get data from a file reader
def get_num(data,date):
	for each in data:
		if date==each[1]:
			return each[3]
	return 0

#calculate cosine similarity of vectors in a matrix
def cos_sim(matx):
	temp=cosine_similarity(matx)
	return temp

#save data to csv file
def savef(path,datas):
	with open(path,'w')as f:
		writer=csv.writer(f)
		writer.writerows(datas)

#data filling
def data_filling(name,odir='/home/fly/cs/data',tdir='/home/fly/cs/stock'):
	calendar=get_date('/home/fly/cs/399300.csv')
	code,ext=os.path.splitext(name)
	print(code+" success")
	mtx=[code]
	path=os.path.join(odir,name)
	with open(path,'r')as f:
		temp=[]
		csvf=csv.reader(f)
		for row in csvf:
			temp.append(row)
		for d in calendar:
			mtx.append(get_num(temp,d))
		return mtx
		#cs=cos_sim(mtx)
		#path=os.path.join('/home/fly/cs/cos300',d+'.csv')
		#path=os.path.join(tdir,name)
		#savef(path,mtx)

#calculate the logarighmic price earnings of stock.
def logearn(stockprice):
	log=[]
	for i in range(0,len(stockprice)-1):
		if stockprice[i]>0 and stockprice[i+1]>0:
			earn=math.log(stockprice[i+1])-math.log(stockprice[i])
			log.append(earn)
		else:
			log.append(0)
	return log

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

#save data to csv file
def save_csv(path,data):
	with open(path,'w')as f:
		csvf=csv.writer(f)
		csvf.writerow(data)

if __name__=="__main__":
	print("Test case.")
	#t=get_num_rows('/home/fly/cs','399300.csv')
	#lst=[]
	#for each in os.listdir('/home/fly/cs/csi300'):
		#temp=get_num_rows('/home/fly/cs/csi300',each)
		#lst.append(temp)
	#stat(lst)
	#data=[]
	codes=[]
	#calendar=[None]+get_date('/home/fly/cs/399300.csv')
	#print(calendar)
	#data.append(calendar)
	for each in os.listdir('/home/fly/cs/data'):
		if get_num_rows('/home/fly/cs/data',each)>2400:
			#fd=data_filling(each)
			#data.append(fd)
			code,ext=os.path.splitext(each)
			codes.append(code)
	save_csv('/home/fly/cs/codes.csv',codes)
	#savef('/home/fly/cs/close.csv',data)
	#calendar=get_date('/home/fly/cs/399300.csv')[1:]
	#tc=[None]+calendar
	#logs=[tc]
	#close=np.loadtxt('/home/fly/cs/close.csv',delimiter=',',skiprows=1,usecols=range(1,2436))
	#code=np.loadtxt('/home/fly/cs/close.csv',delimiter=',',skiprows=1,usecols=0,dtype=str)
	#print("get stock close price success")
	#for i in range(0,len(close)):
		#log=logearn(close[i])
		#tl=[code[i]]+log
		#logs.append(tl)
	#savef('/home/fly/cs/logearn.csv',logs)
	#print("calclate logearn success")
	#loge=np.loadtxt('/home/fly/cs/logearn.csv',delimiter=',',skiprows=1,usecols=range(1,2435))
	#for d in range(0,len(calendar),5):
		#temp=[]
		#if d+10<len(calendar):
			#for row in loge:
				#if 0 not in row[d:d+10]:
					#temp.append(row[d:d+10])
				#else:
					#temp.append([0,0,0,0,0,0,0,0,0,0])
			#pearson=np.corrcoef(temp)
			#path=os.path.join('/home/fly/cs/time',calendar[d+5]+'.csv')
			#savef(path,pearson)
			#print("calculate "+calendar[d+5]+" pearson success")
	print("success")

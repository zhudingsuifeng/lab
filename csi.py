#!/usr/bin/env python
#coding=utf-8
"""
Create on Fri Apr 13 09:38:48 2018
@author: fly
Get CSI300 data.
numpy1.13.1
pandas0.20.3
skimage0.13.0
"""
import os
import csv
import sys
import numpy as np
import tushare as ts
import matplotlib.pyplot as plt
import warnings as wn
import pandas as pd

#get the stock data corresponding to the stock code
def get_stock_data(stock_code,stockdir):
	#stock=ts.get_hist_data(stock_code)    #get latest three years stock historical data
	#stock=ts.get_h_data(stock_code)       #get latest one year historical stock data
	stock=ts.get_k_data(stock_code,start='2006-12-29',end='2016-12-30')        #get all historical stock data,seven columns
	#stock=ts.get_h_data(stock_code,start='2005-04-08')      #get stock data with fifteen columns
	if not os.path.exists(stockdir):
		os.mkdir(stockdir)
	path=os.path.join(stockdir,stock_code+'.csv')
	try:
		stock.to_csv(path)
	except Exception as e:
		return stock_code

if __name__=="__main__":
	wn.filterwarnings("ignore")
	reload(sys)
	sys.setdefaultencoding('utf-8')
	csi300dir='/home/fly/cs/csi300'
	csi500dir='/home/fly/cs/csi500'
	print("Test case.")
	errorstock=[]
	csi300=ts.get_hs300s()  #get stock concept classified.
	csi300_code=np.array(csi300)[:,1]
	for each in csi300_code:
		print(each +" success")
		temp=get_stock_data(each,csi300dir)
		errorstock.append(temp)

	errorstock.append("-----------------")

	csi500=ts.get_zz500s()
	csi500_code=np.array(csi500)[:,1]
	for each in csi500_code:
		print(each+" success")
		temp=get_stock_data(each,csi500dir)
		errorstock.append(temp)
	
	get_stock_data('399300','/home/fly/cs')
	get_stock_data('000905','/home/fly/cs')
	print(errorstock)
	print("success")

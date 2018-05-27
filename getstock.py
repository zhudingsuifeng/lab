#!/usr/bin/env python
#coding=utf-8
"""
Create on Tue Dec 19 15:45:06 2017
@author: fly
Get stock industry.
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

#get stock code
def get_stock_code(fname):
	stock=[]
	with open(fname,'r')as f:
		csvf=csv.reader(f)
		for row in csvf:
			stock.append(row[1])
	return stock[1:]

#get stock data corresponding to the sotck code
def get_stock_data(stock_code,stockdir):
	stock=ts.get_k_data(stock_code,start='2006-08-04',end='2016-08-05')
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
	print("Test case.")
	stockdir='/home/fly/cs/stock'
	errorstock=[]
	#stock=ts.get_today_all()    #get stock real-time quotes and get stock code from it
	#stock.to_csv('/home/fly/cs/allstock.csv')
	stockcode=get_stock_code('/home/fly/cs/allstock.csv')
	for each in stockcode:
		print(each+" success")
		temp=get_stock_data(each,stockdir)
		errorstock.append(temp)
	
	get_stock_data('399300','/home/fly/cs')
	get_stock_data('000905','/home/fly/cs')
	print(errorstock)
    #stock=ts.get_stock_basics()    #get stock basic information.
    #stock=ts.get_cashflow_data(2014,3)   #get 2014,3 cashflow. 
    #stock=ts.get_industry_classified()  #get stock industry classified
    #stock=ts.get_concept_classified()  #get stock concept classified.
    #stock=ts.get_area_classified()   #get stock area classified.
    #stock.to_csv(datadir)     #save stock industry classified to csv file.
	print("success")

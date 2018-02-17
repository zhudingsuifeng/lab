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

if __name__=="__main__":
    wn.filterwarnings("ignore")
    reload(sys)
    sys.setdefaultencoding('utf-8')
    print("Test case.")
    datadir='/home/fly/mygit/data/stock/concept.csv'
    #stock=ts.get_stock_basics()    #get stock basic information.
    #stock=ts.get_cashflow_data(2014,3)   #get 2014,3 cashflow. 
    #stock=ts.get_industry_classified()  #get stock industry classified
    stock=ts.get_concept_classified()  #get stock concept classified.
    #stock=ts.get_area_classified()   #get stock area classified.
    stock.to_csv(datadir)     #save stock industry classified to csv file.
    print("success")

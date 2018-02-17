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

#get stock corresponding line
def getline(code,stocklist):
    for i in range(0,len(stocklist)):
	if code==stocklist[i]:      #stock name---------------------------------------------------
	    return i

#copy the file to the specified directory.
def copyfile(code,industry):
    imgdir='/home/fly/mygit/images/mixing/chcltr/gasf'
    datadir='/home/fly/code/dataset'
    imgoutdir='/home/fly/mygit/filter/images'
    stockdir='/home/fly/mygit/filter/stock'
    classdir='/home/fly/mygit/filter/class'    #-------------------------------
    source=os.path.join(imgdir,code+'.png')
    target=os.path.join(imgoutdir,code+'.png')
    #classify=os.path.join(classdir,industry,'images',code+'.png')
    if os.path.isfile(source):
	os.system("cp %s %s" % (source,target))
    #os.system("cp %s %s" % (source,classify))
    source=os.path.join(datadir,code+'.csv')
    target=os.path.join(stockdir,code+'.csv')
    #classify=os.path.join(classdir,industry,'stock',code+'.csv')
    if os.path.isfile(source):
	os.system("cp %s %s" % (source,target))
    #os.system("cp %s %s" % (source,classify))
    

#stock plate filter
def platefilter(industry):
    codelist=[]
    with open("/home/fly/mygit/data/stock/industry.csv","r") as csvf: #open industry.csv file.
	reader=csv.reader(csvf)
	for line in reader:
	    if line[3]==industry:
		codelist.append(line[1])
    csvf.close()
    return codelist   #return the industry code list


if __name__=="__main__":
    print("Test case.")
    industrylist=[['金融行业','financial'],['有色金属','metal'],['煤炭行业','coal'],['电力行业','electricity'],['钢铁行业','steel'],['酿酒行业','brewing'],['家电行业','appliances'],['生物制药','biopharmaceutical'],['电子信息','digital'],['化纤行业','chemical']]

    #filter stock
    for each in industrylist:
	codelist=platefilter(each[0])
	print("get industry stock code success")
	for i in range(0,len(codelist)):
	    copyfile(codelist[i],each[1])
    	print("copy "+each[0]+" file success")
    print("success")

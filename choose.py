#!/usr/bin/env python
#coding=utf-8
"""
Create on Mon Oct 30 19:14:25 2017
Modify on Tue Apr 24 10:00:45 2018
@author:fly
Pretreatment stock data.Frequent opening of files results in less efficiency.
"""
import os
import csv
import numpy as np

#get stock attribute data with specified length
def get_attr(seq,l):
    return seq[0:l:-1]

#get specified column from file
def get_col(path,c,l):
    with open(path) as csvf:
	reader=csv.reader(csvf)
	reader.next()
	col=map(float,[row[c] for row in reader])
    csvf.close()
    return col[l-1::-1]

#save matrix data to specified file
def save_file(name,data,savedir='/home/fly/hs/interdata'):
    path=os.path.join(savedir,name)
    np.savetxt(path,data,delimiter=',')

if __name__=='__main__':

    mydir='/home/fly/hs/data'
    datadir='/home/fly/hs/interdata'
    filelist=os.listdir(mydir)
    change=[]
    lclose=[]
    close=[]
    trade=[]
    turnover=[]
    for each in filelist:
    	path=os.path.join(mydir,each)
    	(shortname,extension)=os.path.splitext(each)    #get shortname and extension from filename
    	change.append(get_col(path,7,100))
	lclose.append(get_col(path,3,101))
    	close.append(get_col(path,3,100))
   	trade.append(get_col(path,5,100))
	turnover.append(get_col(path,14,100))
    	print(shortname+" success!")
    	#break
    save_file("change.csv",change)
    save_file("lclose.csv",lclose)
    save_file("close.csv",close)
    save_file("trade.csv",trade)
    save_file("turnover.csv",turnover)
    print("success") 

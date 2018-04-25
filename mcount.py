#!/usr/bin/env python
#coding=utf-8
"""
Create on Fri Apr 20 10:40:05 2018
@author: fly
matrix operations
"""
import os
import csv
import numpy as np
import heapq

#get the stock code 
def stock_code(data):
    stockcode=[]
    with open(data,"r") as csvf:
	reader=csv.reader(csvf)
	for line in reader:
	    for i in range(0,len(line)):
		stockcode.append(line[i])
	    break
    return stockcode

#save list to csv file
def save_csv(data,path):
    with open(path,"w") as csvf:
	writer=csv.writer(csvf)
	writer.writerows(data)
    csvf.close()

#The ten largest values in the matrix
def largest(matrix,num,code):
    d={}
    index=[]
    for i in range(0,len(matrix)):
	for j in range(i+1,len(matrix[0])):
	    d[code[i]+','+code[j]]=matrix[i][j]
    temp=sorted(d.items(),key=lambda x:x[1],reverse=True)
    n=0
    for each in temp:
	n+=1
	if n>num:
	    break
	index.append([each[0]])
    return index

#matrix difference
def difference(ma,mb,num,code):
    ma=np.matrix(ma)
    mb=np.matrix(mb)
    mm=ma-mb
    mm=np.array(mm)
    return largest(mm,num,code)

if __name__=="__main__":
    stockcode=stock_code('/home/fly/hs/interdata/stockcode.csv')
    pearson=np.loadtxt('/home/fly/hs/interdata/pearson.csv',delimiter=",")
    cssim=np.loadtxt('/home/fly/hs/interdata/cgasfssim.csv',delimiter=",")
    lssim=np.loadtxt('/home/fly/hs/interdata/lgasfssim.csv',delimiter=",")
    save_csv(largest(pearson,10,stockcode),'/home/fly/hs/interdata/pl10.csv')
    save_csv(largest(cssim,10,stockcode),'/home/fly/hs/interdata/icl10.csv')
    save_csv(largest(lssim,10,stockcode),'/home/fly/hs/interdata/ill10.csv')
    save_csv(difference(pearson,cssim,10,stockcode),'/home/fly/hs/interdata/dcl10.csv')
    save_csv(difference(pearson,lssim,10,stockcode),'/home/fly/hs/interdata/dll10.csv')
    save_csv(difference(lssim,cssim,10,stockcode),'/home/fly/hs/interdata/diil10.csv')
    
    print("success")

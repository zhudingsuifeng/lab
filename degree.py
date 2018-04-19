#!/usr/bin/env python
##!/data/home/xjsjxly/fly/python/bin/python2.7 
#coding=utf-8
"""
Created on Mon Jan 8 19:24:16 2018
Modify on Mon Mar 19 15:40:30 2018
@author: fly
Construct complex network and analysis the network.
datadir:/data/home/xjsjxly/fly/result/data
imagedir:/data/home/xjsjxly/fly/result/images
"""
import os
import sys
import time
import csv
import tushare as ts
import numpy as np
import networkx as nx  #complex network tool
import igraph     #another complex network tool
import community
import matplotlib.pyplot as plt
import scipy.stats as stats

industry=ts.get_industry_classified()
area=ts.get_area_classified()
concept=ts.get_concept_classified()

#get stock code from file
def getstockcode(data):
    stockcode=[]
    with open(data,"r") as csvfile:
	reader=csv.reader(csvfile)
	for line in reader:
	    for i in range(0,len(line)):
		stockcode.append(line[i])
	    break
    csvfile.close()
    return stockcode

#draw a complex network with networkx
def xdrawnet(codes,similarity,threshold):
    fig=plt.figure(figsize=(10,10),dpi=60)
    G=nx.Graph()
    for code in codes:
	G.add_node(code)
    for start in range(0,len(codes)):
	for end in range((start+1),len(codes)):
	    if abs(similarity[start][end])>=threshold:
		G.add_edge(codes[start],codes[end])
    plt.close()
    return G

#
def get_name(code):
    with open("/home/fly/hs/interdata/hs.csv","r") as csvf:
	reader=csv.reader(csvf)
	for line in reader:
	    if code==line[2]:
		return line[3]
    csvf.close()

#
def get_industry(code):
    with open("/home/fly/hs/interdata/industry.csv","r") as csvf:
	reader=csv.reader(csvf)
    	for line in reader:
	    if code==line[1]:
	    	return line[3]
    csvf.close()

#
def get_area(code):
    with open("/home/fly/hs/interdata/area.csv","r") as csvf:
	reader=csv.reader(csvf)
    	for line in reader:
	    if code==line[1]:
	    	return line[3]
    csvf.close()

#
def get_concept(code):
    with open("/home/fly/hs/interdata/concept.csv","r") as csvf:
	reader=csv.reader(csvf)
    	for line in reader:
	    if code==line[1]:
	    	return line[3]
    csvf.close()

#get edge list of the network
def get_degree(stockcode,pearson,imgssim,industry,area,concept):
    savedir='/home/fly/hs/degree'
    #build a network base on thresholds
    for threshold in np.arange(0.2,0.5,0.01):
	pd=[]
	ppath=os.path.join(savedir,"pd"+str(threshold)+".csv")
	pg=xdrawnet(stockcode,pearson,threshold) #dir-----
	for each in sorted(pg.degree(),key=lambda x: x[1],reverse=True):
	    pd.append([each[0],get_name(each[0]),get_industry(each[0]),get_area(each[0]),get_concept(each[0]),each[1]])
	save_csv(pd,ppath)
		
	idg=[]
	ipath=os.path.join(savedir,"id"+str(threshold)+".csv")
	ig=xdrawnet(stockcode,imgssim,threshold)
	for each in sorted(ig.degree(),key=lambda x: x[1],reverse=True):
	    idg.append([each[0],get_name(each[0]),get_industry(each[0]),get_area(each[0]),get_concept(each[0]),each[1]])
	save_csv(idg,ipath)
	
	print(str(threshold)+" success")	
	#break

#save list to csv file
def save_csv(data,path):
    with open(path,"w") as csvf:
	writer=csv.writer(csvf)
	writer.writerows(data)
    csvf.close()

if __name__=="__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    #ts.get_industry_classified()
    stockcode=getstockcode('/home/fly/hs/interdata/stockcode.csv')
    #load csv file.
    imgssim=np.loadtxt('/home/fly/hs/interdata/gasfssim.csv',delimiter=",")
    pearson=np.loadtxt('/home/fly/hs/interdata/pearson.csv',delimiter=",")
    get_degree(stockcode,pearson,imgssim,industry,area,concept)
    print("success")

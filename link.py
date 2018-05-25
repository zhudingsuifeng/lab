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
import time
import csv
import numpy as np
import networkx as nx  #complex network tool
import igraph     #another complex network tool
import community
import matplotlib.pyplot as plt
import scipy.stats as stats

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

#get edge list of the network
def get_edge(stockcode,pearson,imgssim):
    savedir='/home/fly/hs/links'
    #build a network base on thresholds
    for threshold in np.arange(0.28,0.33,0.01):
	'''
	ped=[]
	ppath=os.path.join(savedir,"p"+str(threshold)+".csv")
	pg=xdrawnet(stockcode,pearson,threshold) #dir-----
	for each in pg.edges():
	    ped.append([each[0],each[1]])
	#print(ppath)
	save_csv(ped,ppath)
	'''
	ied=[]
	ipath=os.path.join(savedir,"i"+str(threshold)+".csv")
	ig=xdrawnet(stockcode,imgssim,threshold) #dir-----
	for each in ig.edges():
	    ied.append([each[0],each[1]])
	save_csv(ied,ipath)
	print(str(threshold)+" success")
	#break

#get weight links
def weight_links(code,sim,path):
    ed=[]
    for start in range(0,len(code)):
	for end in range((start+1),len(code)):
	    ed.append([code[start],code[end],sim[start][end]])
    save_csv(ed,path)

#save list to csv file
def save_csv(data,path):
    with open(path,"w") as csvf:
	writer=csv.writer(csvf)
	writer.writerows(data)
    csvf.close()

if __name__=="__main__":
    stockcode=getstockcode('/home/fly/hs/interdata/stockcode.csv')
    #load csv file.
    cssim=np.loadtxt('/home/fly/hs/interdata/cgasfssim.csv',delimiter=",")
    lssim=np.loadtxt('/home/fly/hs/interdata/lgasfssim.csv',delimiter=",")
    pearson=np.loadtxt('/home/fly/hs/interdata/pearson.csv',delimiter=",")
    get_edge(stockcode,pearson,cssim)
    #weight_links(stockcode,pearson,'/home/fly/hs/links/pearson.csv')
    #weight_links(stockcode,cssim,'/home/fly/hs/links/cssim.csv')
    #weight_links(stockcode,lssim,'/home/fly/hs/links/lssim.csv')
    print("success")

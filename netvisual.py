#!/usr/bin/env python 
#coding=utf-8
"""
Created on Mon Jan 8 19:24:16 2018
Modify on Mon Mar 19 15:40:30 2018
@author: fly
Construct complex network and analysis the network.
datadir:/home/fly/data/compare
imagedir:/home/fly/images/compare
"""
import os
import csv
import math
import numpy as np
import networkx as nx  #complex network tool
import community
import matplotlib.pyplot as plt
import scipy.stats as stats

#get stock code from file
def getstockcode(datadir):
    stockcode=[]
    filelist=os.listdir(datadir)
    for each in filelist:
	(code,ext)=os.path.splitext(each)
	stockcode.append(code)
    return stockcode

#dictionary to array
def dicttoarray(dictionary):
    data=[]
    for each in dictionary.items():
	data.append(each)
    return np.array(data)

#draw a complex network with networkx
def xdrawnet(codes,similarity,threshold,savedir,title):
    g=nx.Graph()
    for i in range(0,len(codes)):
	g.add_node(codes[i])
    for start in range(0,len(codes)):
	for end in range((start+1),len(codes)):
	    if abs(similarity[start][end])>=threshold:
		g.add_edge(codes[start],codes[end])
    #g=nx.barabasi_albert_graph(600,6)
    part=community.best_partition(g)
    '''
    temp=[]
    for parta in sorted(dicttoarray(part),key=lambda x : x[1],reverse=True):
    	temp.append(parta[0])

    g=nx.Graph()
    for i in temp:
	g.add_node(temp[i])
    for (u,v) in G.edges():
	g.add_edge(u,v)
    '''
    fig=plt.figure(figsize=(10,10),dpi=100)
    options={
    'edge_color':'grey',
    'linewidths':0,
    'width':0.6,
    'alpha':0.8,
    }   #set networks properties
    nsize=[(200+g.degree(v)*20) for v in g]
    ncolor=[part.get(v)for v in g]
    #nx.draw_spring(g,node_size=nsize,node_color=ncolor,with_labels=False,**options)
    #nx.draw(g,nx.spring_layout(g),node_size=nsize,node_color=ncolor,with_labels=False,**options)
    nx.draw(g,nx.spring_layout(g,k=16/math.sqrt(g.number_of_nodes())),node_size=nsize,node_color=ncolor,with_labels=False,**options)

    netpath=os.path.join(savedir,title+str(threshold)+'xnet.png')
    if not os.path.exists(savedir):         #if directory is not exsits ,make it.
	os.mkdir(savedir)
    plt.savefig(netpath)
    plt.close()
    #return G

if __name__=="__main__":
    stockcode=getstockcode('/home/fly/hs/data')
    imgdir='/home/fly/hs/result'
    #load csv file.
    imgssim=np.loadtxt('/home/fly/hs/interdata/gasfssim.csv',delimiter=",")
    pearson=np.loadtxt('/home/fly/hs/interdata/pearson.csv',delimiter=",")
    for threshold in np.arange(0.48,0.66,0.01):
	xdrawnet(stockcode,pearson,threshold,imgdir,'pearson')
	#xdrawnet(stockcode,imgssim,threshold,imgdir,'imgssim')
    print("success")

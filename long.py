#!/usr/bin/env python
#coding=utf-8
"""
Created on Mon Jan 8 19:24:16 2018
Modify on Mon Mar 19 15:40:30 2018
@author: fly
Construct complex network and analysis the network.
"""
import os
import time
import csv
import numpy as np
import networkx as nx  #complex network tool
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
def xdrawnet(codes,sim,threshold):
    G=nx.Graph()
    for code in codes:
	G.add_node(code)
    for start in range(0,len(codes)):
	for end in range((start+1),len(codes)):
	    if abs(sim[start][end])>=threshold:
		G.add_edge(codes[start],codes[end])
    return G

#statistics network characteristics and save it as plot picture.
def line_plot(data,savedir,title,feature,xl,yl):
    fig=plt.figure(figsize=(10,10),dpi=60)
    plt.title(title,{'fontsize':30})
    plt.plot(data[:,0],data[:,1],color='purple',linewidth=2)
    plt.grid(True,linestyle="--",color="gray",linewidth="1")
    plt.tick_params(labelsize=12)
    plt.xlabel(xl,fontsize=20)
    plt.ylabel(yl,fontsize=20)
    path=os.path.join(savedir,title+feature+'.png')
    if not os.path.exists(savedir):
		os.mkdir(savedir)
    plt.savefig(path)
    plt.close()

#combine two line plot
def comp_plot(de,dc,df,dd,savedir,xl,yl):
    fig=plt.figure(figsize=(10,10),dpi=60)
    #plt.title(feature,{'fontsize':30})
    plt.plot(de[:,0],de[:,1],color='dodgerblue',linewidth=2,label='efficiency')
    plt.plot(dc[:,0],dc[:,1],color='limegreen',linewidth=2,label='cluster')
    plt.plot(df[:,0],df[:,1],color='orangered',linewidth=2,label='fraction')
    plt.plot(dd[:,0],dd[:,1],color='purple',linewidth=2,label='density')
    plt.grid(True,linestyle="--",color="gray",linewidth="1")
    plt.tick_params(labelsize=20)
    plt.xlabel(xl,fontsize=25)
    plt.ylabel(yl,fontsize=25)
    plt.legend(loc=1,fontsize=25)
    path=os.path.join(savedir,'threshold.png')
    if not os.path.exists(savedir):
		os.mkdir(savedir)
    plt.savefig(path)
    plt.close()

#dictionary to list
def dicttolist(dictionary):
    data=[]
    for each in dictionary.items():
	data.append(each)
    return data
	

#analysis of the network
def net(stockcode,sim,title):
    datadir='/home/fly/hs/'
    imgdir='/home/fly/hs/long'
    sub=[]              #number of maximum connected subgraph nodes
    cluster=[]
    efficiency=[]
    density=[]
    paverage_node_connectivity=[]

    #build a network base on thresholds
    for threshold in np.arange(0.01,0.86,0.02):
		#draw network from similarity matrix
		g=xdrawnet(stockcode,sim,threshold) #dir-----
		#large subgraph nodes
		sub.append([threshold,float(len(max(nx.connected_components(g),key=len)))/len(g.nodes)])
		#the average clustering coefficient
		cluster.append([threshold,nx.average_clustering(g)])
		#the average global efficiency of the graph.
		efficiency.append([threshold,nx.global_efficiency(g)])
		#density of a graph
		density.append([threshold,nx.density(g)])                      #return the density of a graph.

    line_plot(np.array(sub),imgdir,title,'fraction','Threshold','value')
    print("subgraph success")
    
    line_plot(np.array(cluster),imgdir,title,'cluster','Threshold','value')
    print("average clustering success")

    line_plot(np.array(efficiency),imgdir,title,'efficiency','Threshold','value')
    print("global efficiency success")
    
    line_plot(np.array(density),imgdir,title,'density','Threshold','value')
    print("density success")
    
    comp_plot(np.array(efficiency),np.array(cluster),np.array(sub),np.array(density),imgdir,'threshold','value')

#ece
def ece(stockcode,pearson,cssim):
    imgdir='/home/fly/hs/long'
    pece=[]
    iece=[]

    #build a network base on thresholds
    for threshold in np.arange(0.01,0.86,0.02):
		#draw network from similarity matrix
		pg=xdrawnet(stockcode,pearson,threshold) #dir-----
		ig=xdrawnet(stockcode,cssim,threshold)
		itemp=float(len(max(nx.connected_components(pg),key=len)))/len(pg.nodes)
		pece.append([threshold,float(len(max(nx.connected_components(pg),key=len)))/len(pg.nodes)-nx.density(pg)])
		iece.append([threshold,itemp-nx.density(ig)])

    ece_plot(np.array(pece),np.array(iece),imgdir,'Threshold','value')
    print("ece success")
    

def ece_plot(dp,di,savedir,xl,yl):
    fig=plt.figure(figsize=(10,10),dpi=60)
    #plt.title(feature,{'fontsize':30})
    plt.plot(dp[:,0],dp[:,1],color='dodgerblue',linewidth=2,label='Traditional')
    plt.plot(di[:,0],di[:,1],color='orangered',linewidth=2,label='Proposed')
    plt.grid(True,linestyle="--",color="gray",linewidth="1")
    plt.tick_params(labelsize=20)
    plt.xlabel(xl,fontsize=25)
    plt.ylabel(yl,fontsize=25)
    plt.legend(loc=1,fontsize=25)
    path=os.path.join(savedir,'Zece.png')
    if not os.path.exists(savedir):
		os.mkdir(savedir)
    plt.savefig(path)
    plt.close()

if __name__=="__main__":
    stockcode=getstockcode('/home/fly/hs/interdata/stockcode.csv')
    cssim=np.loadtxt('/home/fly/hs/interdata/cgasfssim.csv',delimiter=",")
    #lssim=np.loadtxt('/home/fly/hs/interdata/lgasfssim.csv',delimiter=",")
    pearson=np.loadtxt('/home/fly/mygit/data/similarity/pearson.csv',delimiter=",")
    net(stockcode,cssim,'cssim')
    #net(stockcode,lssim,'lssim')
    #net(stockcode,pearson,'pearson')
    ece(stockcode,pearson,cssim)
    print("success")

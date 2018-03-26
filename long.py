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
def xdrawnet(codes,similarity,threshold,imdir,title):
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

#statistics network characteristics and save it as a file.
def featurefile(data,savedir,threshold,title,feature):      #data is a dictionary.
    path=os.path.join(savedir,title+feature+str(threshold)+".csv")
    if not os.path.exists(savedir):
	os.mkdir(savedir)
    csvf=open(path,"a")    #write data to csv file by row.
    writer=csv.writer(csvf)
    for node in sorted(data,key=lambda x : x[1] , reverse=True):  #sort the node with degree
	writer.writerow(node)
    csvf.close()

#statistics network characteristics and save it as plot picture.
def featureplot(data,savedir,title,feature,xl,yl):
    fig=plt.figure(figsize=(10,10),dpi=60)
    plt.title(title,{'fontsize':30})
    plt.plot(data[:,0],data[:,1],color='purple',linewidth=2)
    plt.grid(True,linestyle="--",color="gray",linewidth="1")
    plt.tick_params(labelsize=12)
    plt.xlabel(xl,fontsize=16)
    plt.ylabel(yl,fontsize=16)
    path=os.path.join(savedir,title+feature+'.png')
    if not os.path.exists(savedir):
	os.mkdir(savedir)
    plt.savefig(path)
    plt.close()

#combine two line plot
def complot(datap,datai,savedir,feature,xl,yl):
    featureplot(datap,savedir,'pearson',feature,xl,yl)
    featureplot(datai,savedir,'imgssim',feature,xl,yl)
    fig=plt.figure(figsize=(10,10),dpi=60)
    plt.title(feature,{'fontsize':30})
    plt.plot(datap[:,0],datap[:,1],color='dodgerblue',linewidth=2)
    plt.plot(datai[:,0],datai[:,1],color='purple',linewidth=2)
    plt.grid(True,linestyle="--",color="gray",linewidth="1")
    plt.tick_params(labelsize=12)
    plt.xlabel(xl,fontsize=16)
    plt.ylabel(yl,fontsize=16)
    path=os.path.join(savedir,feature+'.png')
    if not os.path.exists(savedir):
	os.mkdir(savedir)
    plt.savefig(path)
    plt.close()

#statistics network characteristics and save it as scatter picture.
def featurescatter(data,savedir,threshold,title,feature,xl,yl):
    fig=plt.figure(figsize=(10,10),dpi=60)
    plt.scatter(data[:,0],data[:,1],alpha=0.3,c='purple')
    plt.title(title,{'fontsize':30})
    plt.tick_params(labelsize=12)
    plt.xlabel(xl,fontsize=16)
    plt.ylabel(yl,fontsize=16)
    path=os.path.join(savedir,title+feature+str(threshold)+'.png')
    if not os.path.exists(savedir):
	os.mkdir(savedir)
    plt.savefig(path)
    plt.close()

#combine two scatter 
def comscatter(datap,datai,threshold,savedir,feature,xl,yl):
    featurescatter(datap,savedir,threshold,'pearson',feature,xl,yl)
    featurescatter(datai,savedir,threshold,'imgssim',feature,xl,yl)
    fig=plt.figure(figsize=(10,10),dpi=60)
    plt.scatter(datap[:,0],datap[:,1],alpha=0.3,c='dodgerblue')
    plt.scatter(datai[:,0],datai[:,1],alpha=0.3,c='purple')
    plt.title(feature,{'fontsize':30})
    plt.tick_params(labelsize=12)
    plt.xlabel(xl,fontsize=16)
    plt.ylabel(yl,fontsize=16)
    path=os.path.join(savedir,feature+str(threshold)+'.png')
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
	
#degree
def degnodes(data):
    redata=[]
    for i in range(0,len(data)):
	redata.append([i,data[i]])
    return redata

#log
def lognodes(data):
    redata=[]
    for i in range(1,len(data)):
	redata.append([np.log(i),np.log(data[i])])
    return redata

#analysis of the network
def netanalysis(stockcode,pearson,imgssim):
    datadir='/home/fly/mygit/data/temp/long'
    imgdir='/home/fly/mygit/images/compare'
    psubnodes=[]              #number of maximum connected subgraph nodes
    isubnodes=[]              #number of maximum connected subgraph nodes

    paverage_clustering=[]
    iaverage_clustering=[]

    pglobal_efficiency=[]
    iglobal_efficiency=[]

    pdensity=[]
    idensity=[]

    paverage_node_connectivity=[]
    iaverage_node_connectivity=[]

    #build a network base on thresholds
    for threshold in np.arange(0.15,0.2,0.01):
	#draw network from similarity matrix
	pg=xdrawnet(stockcode,pearson,threshold,imgdir,'pearson') #dir-----
	ig=xdrawnet(stockcode,imgssim,threshold,imgdir,'imgssim')

	#large subgraph nodes
	psubnodes.append([threshold,len(max(nx.connected_components(pg),key=len))])
	isubnodes.append([threshold,len(max(nx.connected_components(ig),key=len))])
	
	#the average clustering coefficient
	paverage_clustering.append([threshold,nx.average_clustering(pg)])
	iaverage_clustering.append([threshold,nx.average_clustering(ig)])

	#the average global efficiency of the graph.
	pglobal_efficiency.append([threshold,nx.global_efficiency(pg)])
	iglobal_efficiency.append([threshold,nx.global_efficiency(ig)])

	#density of a graph
	pdensity.append([threshold,nx.density(pg)])                      #return the density of a graph.
	idensity.append([threshold,nx.density(ig)])                      #return the density of a graph.


    #complot(np.array(psubnodes),np.array(isubnodes),imgdir,'largesubgraph','Threshold','nodes')
    savefile(datadir,"plargesubgraph.csv",psubnodes)
    savefile(datadir,"ilargesubgraph.csv",isubnodes)
    print("subgraph success")
    
    #complot(np.array(paverage_clustering),np.array(iaverage_clustering),imgdir,'averagecluster','Threshold','value')
    savefile(datadir,"paverage_clustering.csv",paverage_clustering)
    savefile(datadir,"iaverage_clustering.csv",iaverage_clustering)
    print("average clustering success")

    #complot(np.array(pglobal_efficiency),np.array(iglobal_efficiency),imgdir,'globalefficiency','Threshold','value')
    savefile(datadir,"pglobalefficiency.csv",pglobal_efficiency)
    savefile(datadir,"iglobalefficiency.csv",iglobal_efficiency)
    print("global efficiency success")
    
    #complot(np.array(pdensity),np.array(idensity),imgdir,'density','Threshold','value')
    savefile(datadir,"pdensity.csv",pdensity)
    savefile(datadir,"idensity.csv",idensity)
    print("density success")

#save data to csv file
def savefile(savedir,filename,data):
    path=os.path.join(savedir,filename)
    if not os.path.exists(savedir):
	os.mkdir(savedir)
    np.savetxt(path,data,delimiter=',')

if __name__=="__main__":
    #stockcode=getstockcode('/data/home/xjsjxly/fly/data/stockcode.csv')
    stockcode=getstockcode('/home/fly/mygit/data/stock/stockcode.csv')
    #load csv file.
    #imgssim=np.loadtxt('/data/home/xjsjxly/fly/data/gasfssim.csv',delimiter=",")
    #pearson=np.loadtxt('/data/home/xjsjxly/fly/data/pearson.csv',delimiter=",")
    imgssim=np.loadtxt('/home/fly/mygit/data/similarity/gasfssim.csv',delimiter=",")
    pearson=np.loadtxt('/home/fly/mygit/data/similarity/pearson.csv',delimiter=",")
    netanalysis(stockcode,pearson,imgssim)
    print("success")

#!/usr/bin/env python 
#coding=utf-8
"""
Created on Mon Dec 4 10:45:39 2017
@author: fly
Construct complex network and analysis the network.
"""
import os
import csv
import numpy as np
import networkx as nx  #complex network tool
import igraph     #another complex network tool
import community
import matplotlib.pyplot as plt
import scipy.stats as stats

#draw a complex network with networkx
def xdrawnet(codes,similarity,threshold,savedir,title):
    fig=plt.figure(figsize=(10,10),dpi=60)
    G=nx.Graph()
    for code in codes:       #add node with code
	G.add_node(code)
    for start in range(0,len(codes)):
	for end in range((start+1),len(codes)):
	    if abs(similarity[start][end])>=threshold:            #set the threshold
		G.add_edge(codes[start],codes[end])
    options={
    'node_color':'purple',
    'node_size':100,
    'edge_color':'grey',
    'linewidths':0,
    'width':0.1,
    'alpha':0.6,
    }   #set networks properties
    #nx.draw(G,with_labels=False,**options)
    #community division
    #part=community.best_partition(G)    #---------------------------------------------------
    #values=[part.get(node) for node in G.nodes()] #-----------------------------------------
    nx.draw_spring(G,with_labels=False,**options)
    netpath=os.path.join(savedir,title+str(threshold)+'xnet.png')
    plt.savefig(netpath)
    plt.close()
    return G

#draw a complex network with igraph
def idrawnet(codes,similarity,threshold,savedir,title):
    fig=plt.figure(figsize=(10,10),dpi=60)
    G=igraph.Graph()         #draw network
    for code in codes:       #add node with code
	G.add_vertex(code)   
    for start in range(0,len(codes)):
	for end in range((start+1),len(codes)):
	    if abs(similarity[start][end])>=threshold:            #set the threshold
		G.add_edge(codes[start],codes[end])          #add edge
    visual_style={}                    #set networks properties
    visual_style["vertex_size"]=8
    visual_style["edge_width"]=0.6 
    visual_style["edge_color"]="grey" 
    visual_style["vertex_color"]="purple" 
    visual_style["bbox"]=(600,600) 
    visual_style["margin"]=30
    netpath=os.path.join(savedir,title+str(threshold)+'inet.png')
    igraph.plot(G,netpath,layout=G.layout("kk"),**visual_style)   #draw network and save file.
    return G

#degree distribution in statistical network with networkx
def xcount(G,savedir,title):
    #draw the degree distribution picture
    fig=plt.figure(figsize=(10,10),dpi=60)
    degree=nx.degree_histogram(G)             #get the degree distribution of graph with networkx.
    node=np.arange(len(degree))
    plt.scatter(node,degree,alpha=0.3,c='purple')
    plt.title('Degree distribution',{'fontsize':36})
    plt.tick_params(labelsize=20)
    plt.xlabel('Degree')
    plt.ylabel('Number of vertices')
    degreepath=os.path.join(savedir,title+str(threshold)+'xd.png')
    plt.savefig(degreepath)
    plt.close()

#degree distribution in statistical network with igraph,realize by myself,a little different to networkx.
def icount(G,savedir,title):
    #draw the degree distribution picture
    fig=plt.figure(figsize=(10,10),dpi=60)
    node=[]
    num=[]
    degree=G.degree()  #get the degree distribution of graph with igraph.
    degree.sort()      #sort list small->big
    deg=set(degree)
    for item in deg:
	node.append(item)
	num.append(degree.count(item))
    plt.scatter(node,num,alpha=0.3,c='purple')
    plt.title('Degree distribution',{'fontsize':36})
    plt.tick_params(labelsize=20)
    plt.xlabel('Degree')
    plt.ylabel('Number of vertices')
    degreepath=os.path.join(savedir,title+str(threshold)+'id.png')
    plt.savefig(degreepath)

if __name__=="__main__":
    codedir='/home/fly/mygit/filter/images'
    netdir='/home/fly/mygit/filter/net'
    codes=[]
    #title='ssim'
    #load csv file.
    #similarity=np.loadtxt('/home/fly/mygit/filter/result/platessim.csv',delimiter=",")
    title='pearson'
    similarity=np.loadtxt('/home/fly/mygit/filter/result/pearson.csv',delimiter=",")
    filelist=os.listdir(codedir)
    for i in range(0,len(filelist)):
	(code,ext)=os.path.splitext(filelist[i])
	codes.append(code)
    #build the network based on thresholds.
    for threshold in np.arange(0.3,0.8,0.1):
	#g=xdrawnet(codes,similarity,threshold,netdir,title)
	g=idrawnet(codes,similarity,threshold,netdir,title)
	print("draw "+str(threshold)+" network success")
	#xcount(g,netdir,title)
	#icount(g,netdir,title)
	#print("calculate "+str(threshold)+" degree distribution success")
	#break
    print("success")

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
import warnings as wn
import matplotlib.pyplot as plt
import scipy.stats as stats

wn.filterwarnings("ignore")
stockdir='/home/fly/code/dataset'
filelist=os.listdir(stockdir)
imagedir='/home/fly/mygit/images/net/ssim'
#thresholds=[0.65,0.64,0.63,0.62,0.6,0.58,0.56,0.54,0.52,0.5]
#load csv file.
similarity=np.loadtxt('/home/fly/mygit/data/similarity/gasfssim.csv',delimiter=",")

#build the network based on thresholds.
for threshold in np.arange(0.1,0.2,0.1):
    #fig=plt.figure(figsize=(10,10),dpi=60)
    #G=nx.Graph()
    G=igraph.Graph()
    name=[]
    for i in range(0,len(filelist)):
	(shortname,extension)=os.path.splitext(filelist[i])
	#G.add_node(shortname)
	G.add_vertex(shortname)
	name.append(shortname)
    for start in range(0,len(filelist)):
	for end in range((start+1),len(filelist)):
	    if similarity[start][end]>=threshold:            #set the threshold
		#print(name[start]+" and "+name[end]+" similarity is "+str(similarity[start][end]))
		#G.add_edge(name[start],name[end])
		G.add_edge(name[start],name[end])
    #options={
    #'node_color':'purple',
    options={
    'node_size':100,
    'edge_color':'grey',
    'linewidths':0,
    'width':0.1,
    'alpha':0.3,
    }   #set networks properties
    visual_style={}
    visual_style["vertex_size"]=10 
    visual_style["edge_width"]=0.6 
    visual_style["edge_color"]="grey" 
    visual_style["vertex_color"]="purple" 
    visual_style["bbox"]=(1000,1000) 
    visual_style["margin"]=30
    #nx.draw(G,with_labels=False,**options)
    
    #print graph information
    #print(G.graph)

    #print node information
    #print(G.nodes(data=True))

    #print edges information
    #print(G.edges())

    #print network transitivity
    #print(nx.transitivity(G))

    #print nodes number
    #print(G.number_of_nodes())

    #print edges number
    #print(G.number_of_edges())

    #print neighbors of node
    #print(G.neighbors())

    #community division
    #part=community.best_partition(G)    #---------------------------------------------------
    #print(part)

    #calculate the modularity
    #mod=community.modularity(part,G)
    #print(mod)

    #values=[part.get(node) for node in G.nodes()] #-----------------------------------------
    #nx.draw_spring(G,node_color=values,with_labels=False,**options)
    imagepath=os.path.join(imagedir,'Threshold'+str(threshold)+'.png')
    igraph.plot(G,imagepath,layout=G.layout("lgl"),**visual_style)
    #imagepath=os.path.join(imagedir,'Threshold'+str(threshold)+'.png')
    #plt.savefig(imagepath)
    #plt.close()
    print('save threshold '+str(threshold)+' success++')

    '''
    #print(nx.degree_histogram(G))
    #draw the degree distribution picture
    fig=plt.figure(figsize=(10,10),dpi=60)
    degree=nx.degree_histogram(G)
    node=np.arange(len(degree))
    plt.scatter(node,degree,alpha=0.3,c='purple')
    plt.title('Degree distribution',{'fontsize':36})
    plt.tick_params(labelsize=20)
    plt.xlabel('Degree')
    plt.ylabel('Number of vertices')
    degreedir=os.path.join(imagedir,'Degree distribution of Threshold '+str(threshold)+'.png')
    plt.savefig(degreedir)
    #plt.show()
    plt.close()
    print("degree distribution")
    '''
    #for node,degree in sorted(G.degree(),key=lambda x : x[1] , reverse=True):  #sort the node with degree 
	#G.degree() return n,d ,n is the node name ,d is the node n's degree
	#print('%s %d' % (node,degree))
    #break

print("success")
#print(sorted(nx.degree(G),key=lambda x : x[1],reverse=True)) #sort data by key
#for key,value in sorted(nx.degree(G),key=lambda x:x[1],reverse=True):#sort dict of the value 

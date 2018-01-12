#!/usr/bin/env python 
#coding=utf-8
"""
Created on Mon Jan 8 19:24:16 2018
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
    for code in codes:
	G.add_node(code)
    for start in range(0,len(codes)):
	for end in range((start+1),len(codes)):
	    if abs(similarity[start][end])>=threshold:
		G.add_edge(codes[start],codes[end])
    options={
    'node_color':'purple',
    'node_size':60,
    'edge_color':'grey',
    'linewidths':0,
    'width':0.1,
    'alpha':0.5,
    }   #set networks properties
    #community division
    nx.draw_spring(G,with_labels=False,**options)
    netpath=os.path.join(savedir,title+str(threshold)+'xnet.png')
    plt.savefig(netpath)
    plt.close()
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

#get degree distribution
def xdegree(G,savedir,title,threshold):
    path=os.path.join(savedir,"degree"+str(threshold)+".csv")
    csvf=open(path,"a")    #write data to csv file by row.
    writer=csv.writer(csvf)
    for node,degree in sorted(G.degree(),key=lambda x : x[1] , reverse=True):  #sort the node with degree
	sortnodes=[]
	sortnodes.append(node)
	sortnodes.append(degree)
	writer.writerow(sortnodes)
    csvf.close()

#get subgraph nodes
def xsubgraph(G):
    subgraph=max(nx.connected_components(G),key=len)
    #nodes=subgraph.number_of_nodes() #get nodes number
    nodes=len(subgraph)
    return nodes

#community division ,calculate the modularity and draw community
def xcommunity(G,savedir,title,threshold):
    part=community.best_partition(G)    #Get the module level when the largest community division
    Q=community.modularity(part,G)
    size=len(set(part.values()))
    fig=plt.figure(figsize=(10,10),dpi=60)
    options={
    'node_size':60,
    'edge_color':'grey',
    'width':0.3,
    'alpha':0.5,
    }   #set networks properties
    pos=nx.spring_layout(G)
    count=0.
    for com in set(part.values()):
	count=count+1.
	list_nodes=[nodes for nodes in part.keys()
			      if part[nodes]==com]
	nx.draw_networkx_nodes(G,pos,list_nodes,node_color=str(count/size),**options)
    nx.draw_spring(G,with_labels=False,**options)
    netpath=os.path.join(savedir,title+str(threshold)+'xcommunity.png')
    plt.savefig(netpath) 
    return Q,size

#save data to csv file
def savefile(savedir,filename,data):
    path=os.path.join(savedir,filename)
    np.savetxt(path,data,delimiter=',')

if __name__=="__main__":
    codedir='/home/fly/mygit/images/mixing/chcltr/gasf'
    netdir='/home/fly/mygit/images/net/status'
    codes=[]
    title='ssim'
    sizes=[]
    communities=[]
    #load csv file.
    similarity=np.loadtxt('/home/fly/mygit/data/similarity/gasfssim.csv',delimiter=",")
    filelist=os.listdir(codedir)
    for i in range(0,len(filelist)):
	(code,ext)=os.path.splitext(filelist[i])
	codes.append(code)
    #build the network based on thresholds.
    for threshold in np.arange(0.14,0.125,-0.01):
	g=xdrawnet(codes,similarity,threshold,netdir,title)    #draw network
	print("draw "+str(threshold)+" network success")
	xcount(g,netdir,title)                                 #statistical degree distribution
	print("calculate "+str(threshold)+" degree distribution success")
	nodes=xsubgraph(g)                                     #largest subgraph nodes
	sizes.append([threshold,nodes])
	#print(sizes)
	print("The largest subgraph has "+str(nodes)+" nodes.")
	xdegree(g,netdir,title,threshold)                      #sort nodes by degree
	print("get a larger degree of node of threshold "+str(threshold))
	q,s=xcommunity(g,netdir,title,threshold)               #calculate module degree and community number
	Q=float("%.3f" % q)
	communities.append([threshold,s,Q])
	#print(communities)
	print(str(threshold)+" net has "+str(s)+" communities and modularity is "+str(q))
	#break
    savefile(netdir,"community.csv",communities)
    print("calculate community success")
    savefile(netdir,"largesubgraph.csv",sizes)
    print("subgraph success")
    print("success")

    #print network transitivity
    #print(nx.transitivity(G))

#!/usr/bin/env python
#coding=utf-8

'''
Created on Sat Jan 13 10:36:47 2018
Analyze network attributes.
@author:fly
'''

import os
import csv
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats 
import pandas as pd
import numpy as np

#get data
def getdata(path):
    data=np.loadtxt(path,delimiter=",")
    return data

#plot a time series line chart and save image to file.
def plots(imagedir,name,series,interval,T):
    plt.figure(figsize=(10,10),dpi=60)
    plt.title(T,{'fontsize':36})   #title(label,fontdict) label is str ,fontdict is dict {key:value},a dictionary controlling the appearance of the title text.
    plt.plot(interval,series,color='purple',linewidth=2)
    plt.grid(True,linestyle="--",color="gray",linewidth="1")
    plt.xlabel('Threshold')
    plt.ylabel('Number of nodes')
    savepath=os.path.join(imagedir,name+".png")
    plt.savefig(savepath)
    plt.close()

if __name__=="__main__":
    subgraph_path='/home/fly/mygit/images/net/largesubgraph.csv'
    community_path='/home/fly/mygit/images/net/community.csv'
    imagedir='/home/fly/mygit/images/net/image'
    subo=getdata(subgraph_path)
    subgraph=subo[::-1]   #obtain the walue of the sequence in reverse order
    plots(imagedir,"largesubgraph",subgraph[:,1],subgraph[:,0],"Node's number of arge subgraph")
    print("draw subgraph nodes success")
    como=getdata(community_path)
    community=como[::-1]
    plots(imagedir,"community",community[:,1],community[:,0],"Community of network")
    plots(imagedir,"modularity",community[:,2],community[:,0],"Modularity of community")
    print("draw community success")
    print("success")


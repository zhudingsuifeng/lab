#!/usr/bin/env python
#coding=utf-8
'''
Created on Sat Jan 13 10:36:47 2018
Analyze network attributes.
@author:fly
'''
import os
import csv
import sys
import math
import networkx as nx
import matplotlib.pyplot as plt
import scipy.stats as stats 
import pandas as pd
import numpy as np

#get date from index
def get_date(path):
	data=np.loadtxt(path,delimiter=',',skiprows=1,usecols=1,dtype=str)
	for i in range(6,len(data),5):
		yield data[i]

#sum
def get_sum(lst):
	temp=0
	for i in lst:
		if np.isnan(i):
			temp=temp
		else:
			temp+=i
	return temp

#get the average similarity
def ave_sim(data):
	avesim=0
	n=0
	for row in data:
		for i in row:
			if not np.isnan(i):
				n+=1
		break
	for row in data:
		avesim+=get_sum(row)	
	return float(avesim-n)/(n*(n-1))

#get hs300 close price
def get_hs(path):
	data=np.loadtxt(path,delimiter=',',skiprows=1,usecols=3)
	for i in range(6,len(data),5):
		yield data[i]/max(data)
	
#get the minimum threshold through minimum spanning tree
def min_thold_tree(codes,sim):
	g=nx.Graph()
	for code in codes:
		g.add_node(code)
	for start in range(0,len(codes)):
		for end in range((start+1),len(codes)):
			if not np.isnan(sim[start][end]):
				g.add_edge(codes[start],codes[end],weight=sim[start][end])
	t=nx.minimum_spanning_tree(g)
	th=min([d for (u,v,d) in t.edges(data=True)])
	return th

#get the minimum threshold through try
def min_thold_try():
	return th

#draw a network with the threshold
def draw_net(codes,sim,th):
	g=nx.Graph()
	for code in codes:
		g.add_node(code)
	for start in range(0,len(codes)):
		for end in range((start+1),len(codes)):
			if not np.isnan(sim[start][end]) and sim[start][end]>=th:
				g.add_edge(codes[start],codes[end])
	return g

#draw a network with weight,put here temporarily
def dw_net(codes,sim):
	g=nx.Graph()
	for code in codes:
		g.add_node(code)
	for start in range(0,len(codes)):
		for end in range((start+1),len(codes)):
			if not np.isnan(sim[start][end]):
				g.add_edge(codes[start],codes[end],weight=sim[start][end])
	return g

#get clustering coefficient
def get_cc(g):
	return nx.average_clustering(g)

#get the degree distribution entropy from a network
def get_de(g):

	return de

#get the shortest average distance of the network
def get_sad(g):
	return np.average_shortest_path_length(g)

#plot a time series line chart and save image to file.
def plots(series,name):
	time=list(get_date('/home/fly/cs/399300.csv'))
	time=[time[i] for i in range(0,len(time),50)]
	s=[i*50 for i in range(0,len(time))]
	plt.figure(figsize=(10,10),dpi=60)
	plt.plot(series,color='purple',linewidth=2,label='test')
	plt.grid(True,linestyle="--",color="gray",linewidth="1")
	plt.xticks(s,time,rotation=30)
	plt.xlabel('time')
	plt.ylabel('value')
	plt.legend(loc=1)
	savepath=os.path.join('/home/fly/cs/result',name+".png")
	plt.savefig(savepath)
	plt.close()

if __name__=="__main__":
	csi300=list(get_hs('/home/fly/cs/399300.csv'))
	#plots(csi300,'csi500')
	csi500=list(get_hs('/home/fly/cs/000905.csv'))
	plots(csi500,'csi500')
	#for each in os.listdir('/home/fly/cs/time'):
		#path=os.path.join('/home/fly/cs/time',each)
		#data=np.loadtxt(path,delimiter=',')
		#print(ave_sim(data))
		#break
	#plots(imagedir,"modularity",community[:,2],community[:,0],"Modularity of community")
	print("success")


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

#get stock code
def get_code(path):
	codes=np.loadtxt(path,delimiter=',',dtype=str)
	return codes

#get the average similarity
def ave_sim(data):
	avesim=0
	n=0
	for row in data:
		avesim+=get_sum(row)
		for i in row:
			if not np.isnan(i):
				n+=1
	if n>0:
		t=n**0.5
		return float(avesim-t)/(t*(t-1))
	else:
		return 0

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
				g.add_edge(codes[start],codes[end],weight=abs(sim[start][end]))
	#t=nx.minimum_spanning_tree(g)
	t=nx.maximum_spanning_tree(g)
	if len(list(t.edges()))>0:
		th=min(map(abs,[d['weight'] for (u,v,d) in t.edges(data=True)]))
		#th=min([d['weight'] for (u,v,d) in t.edges(data=True)])
	else:
		th=0
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
			if not np.isnan(sim[start][end]) and abs(sim[start][end])>=th:
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
	hist=nx.degree_histogram(g)
	de=-sum([(float(i)/sum(hist))*math.log(float(i)/sum(hist)) for i in hist if i>0])
	return de

#get the shortest average distance of the network
def get_sad(g):
	return sum([nx.average_shortest_path_length(g.subgraph(c)) for c in nx.connected_components(g)])

#plot a time series line chart and save image to file.
def plots(series,name,l):
	time=list(get_date('/home/fly/cs/399300.csv'))
	time=[time[i] for i in range(0,len(time),50)]
	s=[i*50 for i in range(0,len(time))]
	plt.figure(figsize=(10,10),dpi=60)
	plt.plot(series,color='purple',linewidth=2,label=l)
	plt.grid(True,linestyle="--",color="gray",linewidth="1")
	plt.xticks(s,time,rotation=30)
	plt.xlabel('time')
	plt.ylabel('value')
	plt.legend(loc=1)
	savepath=os.path.join('/home/fly/cs/result',name+".png")
	plt.savefig(savepath)
	plt.close()

#save data to csv file
def save_csv(path,data):
	with open(path,'w')as f:
		csvf=csv.writer(f)
		csvf.writerow(data)
if __name__=="__main__":
	csi300=list(get_hs('/home/fly/cs/399300.csv'))
	plots(csi300,'csi300','csi300')
	csi500=list(get_hs('/home/fly/cs/000905.csv'))
	plots(csi500,'csi500','csi500')
	cc=[]
	de=[]
	sad=[]
	sim=[]
	for each in os.listdir('/home/fly/cs/time'):
		path=os.path.join('/home/fly/cs/time',each)
		data=np.loadtxt(path,delimiter=',')
		th=ave_sim(data)
		sim.append(th)
		codes=get_code('/home/fly/cs/codes.csv')
		v=min_thold_tree(codes,data)
		#print(v)
		g=draw_net(codes,data,th)
		cc.append(get_cc(g))
		#sad.append(get_sad(g))
		de.append(get_de(g))
		print(each+" success")
		#break
	plots(sim,'average_sim','ave_sim_ave')
	save_csv('/home/fly/cs/result/average_sim.csv',sim)
	plots(cc,'average_cc','ave_cc_ave')
	save_csv('/home/fly/cs/result/average_cc.csv',cc)
	plots(de,'degree_entropy','dentropy_ave')
	save_csv('/home/fly/cs/result/degree_entropy.csv',de)
	print("success")


#!/usr/bin/env python
#coding=utf-8
"""
Created on Wed Sep 13 15:23:18 CST 2017
@author:fly
Encoding the change price to compare the trend of the ups and downs.
"""

import os 
import csv
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import warnings as wn

wn.filterwarnings("ignore")
G=nx.Graph()                        #create a graph object
mydir='../data'
imdir='../images'
path=os.path.join(mydir,"changefile.csv")
newpath=os.path.join(mydir,"encoding.csv")
finalpath=os.path.join(mydir,"final1.csv")
encoding=[]                         #save the coding
final=[]
with open(path) as csvf:            #open file  
    reader=csv.reader(csvf)         #read file
    i=0
    for row in reader:
	encoding.append([float(row[0])])
	final.append([row[0]])
	G.add_node(row[0])                     #add nodes to network
	for col in range(1,101):
	    if(float(row[col])==0):
		encoding[i].append(0)          #export data to updown[][]
	    elif(float(row[col])>0):
		encoding[i].append(1)
	    else: 
		encoding[i].append(-1)
	i+=1
    csvf.close()
#print(encoding)

#np.savetxt(newpath,encoding,delimiter=',')          #save matrix to csv file

for online in range(0,2918):
    for offline in range(0,2918):
	interarray=list(abs(x-y) for x,y in zip(encoding[online][1:],encoding[offline][1:]))
	#print(interarray)
	result=float(sum(interarray))/100
	#print(result)
	final[online].append(result)
	if(result<0.8):
	    G.add_edge(final[online][0],final[offline][0])
	    print("add edge success!!")
#np.savetxt(finalpath,final,delimiter=',')           #use this command to save csv file only use the float data

#plt.figure()
#plt.plot(final[0][1:],linewidth=0.1)
#plt.savefig("../images/show.png",dpi=1000)
#plt.show()
nx.draw(G,width=0.1,with_labels=False,node_size=6)
plt.savefig("../images/net.png")
plt.show()

print("success")

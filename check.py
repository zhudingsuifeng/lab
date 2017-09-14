#!/usr/bin/env python
#coding=utf-8
"""
Created on Tue Sep 12 20:53:33 CST 2017
Modified on Wed Sep 13 08:36:51 CST 2017
@author:fly
calculate discrete point area:Trapezoidal quadrature formula.
"""

import os 
import csv
import matplotlib.pyplot as plt
import numpy as np
import warnings as wn

wn.filterwarnings("ignore")
mydir='../data'
imdir='../images'
path=os.path.join(mydir,"changefile.csv")
with open(path) as csvf:            #open file  
    reader=csv.reader(csvf)         #read file
    updown=[]
    stock=[]
    i=0
    for row in reader:
	updown.append([])
	stock.append(row[0])
	for col in range(1,101):
	    updown[i].append(float(row[col]))#export data to updown[][]
	i+=1
    csvf.close()

su=0
for i in range(0,100):
    num=abs(updown[0][i]-updown[1][i])
    su+=num
    print(num)
print(su)

'''
arr=[]
for online in range(0,2918):
    #for offline in range((online+1),2918):
    su=0
    for i in range(0,100):
	su+=abs(updown[0][i]-updown[online][i])
	#print(updown[0][i])
    arr.append(su)
print(arr)

plt.figure()
plt.plot(arr)
impath=os.path.join(imdir,"area.png")
plt.savefig(impath)
plt.show()
plt.close()
'''
print("success")

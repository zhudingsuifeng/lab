#!/usr/bin/env python
#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.cluster   import KMeans  
import math
import random  
  
#生成两个高斯分布训练样本用于测试  
#第一类样本类  
mean1 = [0, 0]  
cov1 = [[1, 0], [0, 1]]  # 协方差矩阵  
x1, y1= np.random.multivariate_normal(mean1, cov1, 100).T    
data=[]  
for x,y in zip(x1,y1):  
    data.append([x,y])  
#第二类样本类  
mean2 = [3,3]  
cov2 = [[1, 0], [0, 1]]  # 协方差矩阵  
x2, y2= np.random.multivariate_normal(mean2, cov2, 100).T  
for x,y in zip(x2,y2):  
    data.append([x,y])  
random.shuffle(data)#打乱数据  
data=np.asarray(data,dtype=np.float32)  
#算法开始  
#计算两两样本之间的权重矩阵,在真正使用场景中，样本很多，可以只计算邻接顶点的权重矩阵  
m,n=data.shape  
distance=np.zeros((m,m),dtype=np.float32)  
for i in range(m):  
    for j in range(m):  
        if i==j:  
            continue  
        dis=sum((data[i]-data[j])**2)  
        distance[i,j]=dis  
#构建归一化拉普拉斯矩阵  
similarity = np.exp(-1.* distance/distance.std()) 
f=np.array(np.zeros([m,m]),dtype=np.float32) 
for i in range(m): #归一化操作
    similarity[i,i]=-sum(similarity[i]) 
    similarity[i]=-similarity[i]
    f[i,i]=1/math.sqrt(similarity[i,i])  
x=np.matrix(similarity)
f=np.matrix(f)
x=f*x*f
similarity=np.array(x)
#计算拉普拉斯矩阵的前k个最小特征值  
[Q,V]=np.linalg.eig(similarity)  
idx = Q.argsort()  
Q = Q[idx]  
V = V[:,idx]  
#前3个最小特征值  
num_clusters =3  
newd=V[:,:3] 
#k均值聚类  
clf = KMeans(n_clusters=num_clusters) #KMeans is a class ,clf is a object.
clf.fit(newd)   #Compute k-means clustering.
#print(clf.cluster_centers_,clf.labels_)
#clf.fit(newd)  
#print(type(clf.labels_),clf.labels_)
#显示结果  
for i in range(data.shape[0]):  
    if clf.labels_[i]==0:  
        plt.plot(data[i,0],data[i,1],'go')  
    elif clf.labels_[i]==1:  
        plt.plot(data[i,0],data[i,1],'ro')  
    elif clf.labels_[i]==2:  
        plt.plot(data[i,0],data[i,1],'yo')  
    elif clf.labels_[i]==3:  
        plt.plot(data[i,0],data[i,1],'bo')  
plt.show()

#!/usr/bin/env python3
#coding=utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def excel_to_list(p,l):
    d=pd.read_excel(p,usecols=l)
    d=np.array(d)
    d=d.tolist()
    return d

p='/home/fly/works/data.xlsx'
plt.figure(figsize=(40,30))
m=Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180)
x=excel_to_list(p,[2])[1:]  #longitude经度latitude纬度
x=[i[0] for i in x]
y=excel_to_list(p,[1])[1:]
y=[i[0] for i in y]
x,y=m(x,y) #m=Basemap() m(x,y,invers=True) True coordinate to longitude and latitude,False longitude and latitude to coordinate
m.scatter(x[74730:89696],y[74730:89696],s=1,color='forestgreen',label='2015')
m.scatter(x[89696:103286],y[89696:103286],s=1,color='gold',label='2016')
m.scatter(x[103286:],y[103286:],s=1,color='crimson',label='2017')
m.drawcoastlines(linewidth=0.2) #draw 板块图
m.drawcountries()  #draw countries in map
#m.drawstates()    #draw states in map
plt.title("worldmap")
plt.legend(loc=2)
#plt.legend(loc=2,fontsize=25)
plt.savefig('/home/fly/works/map.png')
print("success")
plt.show()

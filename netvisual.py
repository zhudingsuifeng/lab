#!/usr/bin/env python 

import tushare as ts
import community
import matplotlib.pyplot as plt
import networkx as nx
import igraph

g=igraph.Graph.GRG(36,0.3)
#print(g)
#fig=plt.figure()
visual_style={}
visual_style["vertex_size"]=6
visual_style["bbox"]=(800,800)
visual_style["edge_width"]=0.3
visual_style["edge_color"]="grey"
visual_style["vertex_color"]="purple"
visual_style["margin"]=10
visual_style["vertex_shape"]="circle"
#visual_style[""]=1
igraph.plot(g,'g.png',layout=g.layout("lgl"),**visual_style)
#igraph.add(fig,g)
#igraph.save(g,'g.png')
#plt.savefig('g.png')
#plt.show()

#G=nx.complete_graph(10)

#part=community.best_partition(G)

#print(part)
#df= ts.get_stock_basics()
#path=os.path.join()
#df.to_csv('/home/fly/mygit/data/stock/basic.csv')
print("success")

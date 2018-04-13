#!/usr/bin/env python
#coding=utf-8
"""
Create on Tue Dec 19 22:09:01 2017
@author: fly
Filter stocks by sector.
numpy1.13.1
pandas0.20.3
"""
import os
import csv
import sys
import numpy as np
import tushare as ts
import pandas as pd

#copy the file to the specified directory.
def copyfile(code):
    inimgdir='/home/fly/mygit/images/mixing/chcltr/gasf'
    outimgdir='/home/fly/hs/images'
    indatadir='/home/fly/Pictures/dataset'
    outdatadir='/home/fly/hs/data'
    source=os.path.join(inimgdir,code+'.png')
    target=os.path.join(outimgdir,code+'.png')
    if os.path.isfile(source):
	os.system("cp %s %s" % (source,target))
    source=os.path.join(indatadir,code+'.csv')
    target=os.path.join(outdatadir,code+'.csv')
    if os.path.isfile(source):
	os.system("cp %s %s" % (source,target))
    print("copy "+code+" success")

if __name__=="__main__":
    print("Test case.")
    csi300=ts.get_hs300s()
    csi300_code=np.array(csi300)[:,1]

    #filter stock
    for each in csi300_code:
	copyfile(each)
    print("success")

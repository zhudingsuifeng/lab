#!/usr/bin/env python
#coding=utf-8
"""
Create on Tue Apr 24 16:39:25 2018
@author:fly
sequence stability test
"""
import os
import csv
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as sts

#get row data from file
def get_row(path):
    data=np.loadtxt(path,delimiter=',')
    return data[0]

#adf
def get_adf(seq):
    seq=np.array(seq)
    result=sts.adfuller(seq,1,regresults=True)
    print(result)

if __name__=='__main__':
    path='/home/fly/hs/interdata/change.csv'
    temp=get_row(path)
    get_adf(temp)
    print("success") 

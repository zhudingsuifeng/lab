#!/usr/bin/env python
#coding=utf-8
"""
Modified on Wed Aug 23 10:17:38 2017
@author:fly
filefilter,to remove the file which closing price is zero.
"""
import os
import csv
import warnings as wn

wn.filterwarnings("ignore")
mydir='../dataset'                                 #dir
delnum=0
filelist=os.listdir(mydir)                      #list all file and dir in mydir
for f in range(0,len(filelist)):
    path=os.path.join(mydir,filelist[f])        #combining dir and filename to form a full path
    if os.path.isfile(path):                    #to judge full path is a file or not
	with open(path) as csvf:                #open the csv files
	    reader=csv.reader(csvf)             #read the text from file object
	    next(reader)
	    change=[]
	    dates=[]
	    for row in reader:                  #traverse the text from csv file
		change.append(row[7])           #get col[7] from csv table,quote change
		dates.append(row[0])
            stocklen=len(change)
#            print(path+str(stocklen))
#            if stocklen>=100 and dates[stocklen-1]=='2017-07-17'or dates[stocklen-1]=='2017-07-18' :                  #len of the stock more than 100 
                #print(dates[stocklen-1]) 
            if stocklen>100:
		#print(stocklen)
#		if dates[stocklen-1]=='2017-07-17' or dates[stocklen-1]=='2017-07-18':
#		    print(dates[stocklen-1])
		if dates[0]=='2017-08-22':
		    print(stocklen)         #print stock length
		    """                     #watch out indented although it is annotate
                    for closep in range((stocklen-100),stocklen):
		        num=float(highs[closep])               #convert char to float,csv file number is float (0.0),can't use int()
		        if num==0:  #from 1 to 100 have number is 0
		            csvf.close()
#		            print(path)
		            os.remove(path)
                            delnum+=1               #python do not support delnum++ ,we can replace with delnum+=1         
			    print(path+" is deleted!")
			    """
		    #break
		else:
		    os.remove(path)
		    delnum+=1
		    print(path+" is deleted!!!")
            else:
                os.remove(path)
                delnum+=1
		print(path+" is deleted!")
print("Data preprocessing is complete!"+str(delnum)+"file is deleted!")

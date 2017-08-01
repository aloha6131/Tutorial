# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np

path  = "D:/spyder-ws/test/"
name  = "Book1.csv"
fpath = path+name

print "fpath=",fpath
df=pd.read_csv(fpath)
trueList = df[ df['isDefect(max10)'] ].index.tolist()

print "----------------------"

def listTxSF6HBr(aa):
#    print "Original aa=",aa
    run=1
    flagSF6=0
    flagHBr=0
    diffLast=0
    
    idxSF6=1
    idxHBr=1
    for i in range(0, len(aa)):
        value=aa[i]
        lastValue=value
        if i>0:
            lastValue=aa[i-1][:3]
        newValue=""
        diffLast=0
        if value!=lastValue:
            diffLast=1      
#        print "Iteration(",i,")=",value,"-- run=",run,", idxSF6=",idxSF6,", idxHBr=",idxHBr,", flagSF6=",flagSF6,", flagHBr=",flagHBr,", diffLast=",diffLast
                         
        if flagSF6==1 and flagHBr==1 and diffLast==1:
            run += 1
            idxSF6 = 1
            idxHBr = 1
            flagSF6=0
            flagHBr=0
            
        if value=='SF6':
            newValue=value+"-"+str(run)+"-"+str(idxSF6)
            idxSF6 += 1
            flagSF6=1
        if value=='HBr':
            newValue=value+"-"+str(run)+"-"+str(idxHBr)
            idxHBr += 1
            flagHBr=1
        
        print "value=",value,", newValue=",newValue
        aa[i]=newValue
#    print "Transfer aa=",aa
    return aa


allList = []
for i in range(0, len(trueList)):
    print "i=",i
    if i==0:
        tmpList=df.loc[0              :trueList[i],'recipe_category'].values
    else:
        tmpList=df.loc[trueList[i-1]+1:trueList[i],'recipe_category'].values  
    
    tmpList=tmpList.tolist()
    tmpList=listTxSF6HBr(tmpList)
    allList.append(tmpList)
    print "*****"

print "= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ="
print "allList=\n",allList


#print df['isDefect(max10)']==True
#print "trueList[0]=",trueList[0]
#print "trueList[1]=",trueList[1]

#print "-------------------------"
#print "df.iloc[:,1:2]=\n",df.iloc[:,1:2]
#print "-------------------------"
#print "df.iloc[0:2,1:2]=\n",df.iloc[0:2,1:2]
#print "-------------------------"
#print df.loc[0:1, ['recipe_category']]  #SF6,HBr
#print df.loc[2:5, ['recipe_category']]  #SF6,SF6,HBr,HBr
#print "-------------------------"

#def reMarkValue(run, vList):
#    for item in vList:
       

#def getDistinctListValue(list1):
#    return list(set(list1))

#aaa=['SF6', 'HBr', 'SF6', 'SF6', 'HBr']
#print "aaa=",aaa
#print "type(aaa)=",type(aaa)
#print "aaa[4]=",aaa[4]
#print "aaa[5]=",aaa[5]

# ['SF6-1-1', 'HBr-1-1', 'SF6-2-1', 'SF6-2-2', 'HBr-2-1']
#aa=['SF6', 'HBr', 'SF6', 'SF6', 'HBr']
#aa=['SF6', 'HBr', 'SF6', 'SF6', 'HBr', 'SF6', 'SF6', 'SF6', 'HBr']
#aa=['SF6', 'HBr', 'HBr', 'SF6', 'HBr', 'SF6', 'HBr']
#bb=list(set(aa))
#print "aa=",aa
#print "bb=",bb
#print type(aa)

#aa[0]=aa[0]+'-1-1'
#print aa
#initialValue = aa[0]
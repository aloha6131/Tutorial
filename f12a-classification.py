# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

##########################################
## 1. Preprocess                        ##
##                                      ##
##########################################
window=3
df = pd.read_excel('F12A-sample.xlsx', )
print df.head()
print df.shape
print "--------------"
#trueList = df[ df['GB']=='B' ].index.tolist()
print "df=\n",df

print "--------------"
for i in range(0,window):
    print i
    colName='prev_cate_'+str(i)
    print colName
    df[colName]=''

for i in range(0, len(df)):
    for j in range(0, window):
        tmpList=df.loc[i-window+j:i-window+j,'recipe_category'].values.tolist()
        tmp='X'
        if len(tmpList)>0:
            tmp=tmpList[0]
        print "i=", i, ", j=", j, ", tmp=",tmp, ", type=",type(tmp)
        df.loc[i, 'prev_cate_'+str(j)] = tmp

print df
print "--------------"

le = LabelEncoder()
for i in range(0,window):
    colName='prev_cate_'+str(i)
    df[colName] = le.fit_transform(df[colName])

df=df[['GB','HBr_Accum','SF6_Accum','ratio','prev_cate_0','prev_cate_1','prev_cate_2']]
print df.head()



def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


df2, targets=encode_target(df, "GB")
features = list(df.columns[1:])
y = df2["Target"]
X = df2[features]





##########################################
## 2. Train model                       ##
##                                      ##
##########################################

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print "X.shape=",X.shape
print "y.shape=",y.shape
print "X_train.shape",X_train.shape
print "y_train.shape",y_train.shape
print "X_test.shape",X_test.shape
print "y_test.shape",y_test.shape

print "---------------"
print "X_train=",X_train
print "y_train=",y_train
print "---------------"

#dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)
print clf

#make predictions
predicted = clf.predict(X_test)
#summarize the fit of the model
print metrics.classification_report(y_test, predicted)
print metrics.confusion_matrix(y_test, predicted)

print "-----------"
scores = cross_val_score(clf, X, y, cv=5)
print scores


#print metrics.classfication_report(expected, predicted)
#print metrics.confusion_matrix(expected, predicted)

##########################################
## Reserve Decision Tree Visulization   ##
##                                      ##
##########################################

'''
import graphviz
import tree
from sklearn.tree import export_graphviz
export_graphviz(clf,
                out_file="tree.dot", 
                class_names=["G", "B"],
                feature_names=features, 
                impurity=False, filled=True)

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)
'''

import graphviz
#import tree
from sklearn.tree import export_graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\graphviz-2.38\\bin'
#print os.environ["PATH"]

#dot_data = export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("f12a-tree")



dot_data = export_graphviz(clf, out_file=None, 
                         feature_names=features,  
                         class_names=targets,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("f12a-tree")

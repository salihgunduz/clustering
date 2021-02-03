# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 18:55:13 2020

@author: sg
"""


# seed the pseudorandom number generator
from random import seed
from random import random
from sklearn.cluster import KMeans
# seed random number generator
seed(1)
print(random(), random(), random())

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import *
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.utils.random import sample_without_replacement
from scipy.cluster.hierarchy import dendrogram, linkage,leaves_list
from matplotlib import pyplot as plt


#load iris data
data_iris=pd.read_csv('data/iris.data')
data_cancer=pd.read_csv('data/breast-cancer-wisconsin.data')
data_yeast=pd.read_csv('data/yeast.data')

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data_iris['Iris-setosa'] = le.fit_transform(data_iris['Iris-setosa'])
X1 = data_iris.iloc[:,0:4]
Y = data_iris.iloc[:,4:5]
X=X1.values.tolist()


# set size of subsample
size=90
n = int(len(data_iris) * size/100)
m=100
K=8

# set alogrithms parameters
parameter_list=[K]
cooMat = np.zeros((len(data_iris),len(data_iris)))
cooDict=[]
idx = np.zeros(size)
N = len(parameter_list)*m

#now just using one algorihm. 
for i in range(len(parameter_list)):#algorithms
    k = parameter_list[i]
    km = KMeans(n_clusters=k, init='random',n_init=10, max_iter=300,random_state=42)
    
    for i in range(m):# m data partitions
        data1 = sub_sampling(data_iris, size)  # shuffle and subsample
        data1X = data1.iloc[:,0:4]     # exculde label
        idx = data1X.index
        km_temp =km.fit_predict(data1X)    #k means predict        
        km_temp = pd.DataFrame(km_temp)
        km_temp['idx'] = idx
        km_temp = km_temp.set_index('idx')
        km_temp1 = np.array(km_temp)
        km_temp1 = km_temp1.reshape(1,n)
        km_temp1 = np.repeat(km_temp1,n,axis=0)
        temp_co = ((km_temp1-km_temp1.T)==0)/(N)
        for tempx in range(len(temp_co)):
            for tempy in range(len(temp_co)):
                 cooMat[idx[tempx],idx[tempy]] += temp_co[tempx,tempy]
                
def evel_stab(cluster):
    return 1

    
SL_mat = cooMat.copy() 
SL_mat = pd.DataFrame(SL_mat)

import seaborn as sns; 

ax = sns.heatmap(cooMat)



# this part how can we set K?
single = AgglomerativeClustering(n_clusters=K, linkage='single')
single.fit_predict(SL_mat)
sl_labels = single.labels_

average = AgglomerativeClustering(n_clusters=K, linkage='average')
average.fit_predict(SL_mat)
av_labels = average.labels_

#test
  

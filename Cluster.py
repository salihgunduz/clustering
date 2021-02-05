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
import itertools


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
K=3
M=1
# set alogrithms parameters
parameter_list=[3, 5, 8]

Cs = []
cooDict=[]
idx = np.zeros(size)


#now just using one algorihm knn. 
for i in range(len(parameter_list)):#algorithms
    Co_assoc = np.zeros((len(data_iris),len(data_iris)))
    k = parameter_list[i]
    km = KMeans(n_clusters=k, init='random',n_init=10, max_iter=300,random_state=42)
    counter = np.zeros((149,149))
    for i in range(m):# m data partitions
        data1 = sub_sampling(data_iris, size)  # shuffle and subsample %90
        data1X = data1.iloc[:,0:4]     # exculde label
        idx = data1X.index
        km_temp =km.fit_predict(data1X)    #k means predict        
        km_temp = pd.DataFrame(km_temp)
        km_temp['idx'] = idx
        km_temp = km_temp.set_index('idx')
        km_temp1 = np.array(km_temp)
        km_temp1 = km_temp1.reshape(1,n)
        km_temp1 = np.repeat(km_temp1,n,axis=0)
        temp_co = ((km_temp1-km_temp1.T)==0)
        for i_counter in range(len(temp_co)):
            for j_counter in range(len(temp_co)):
                counter[idx[i_counter],idx[j_counter]]+= 1
        for tempx in range(len(temp_co)):
            for tempy in range(len(temp_co)):
                 Co_assoc[idx[tempx],idx[tempy]] += temp_co[tempx,tempy]
    Co_assoc /=  counter            
    dist_mat = Co_assoc.copy() 
    dist_mat = pd.DataFrame(dist_mat)
    dist_mat = 1 - dist_mat  
    # this part how can we set K?
    single = AgglomerativeClustering(n_clusters=K, linkage='single')
    single.fit_predict(dist_mat)
    
    sl_labels = single.labels_
    sl_labels = pd.DataFrame(sl_labels)
    unique_sl_labels = np.unique(sl_labels)
    sl_labels['idx'] = sl_labels.index
    unique_sl_labels = pd.DataFrame(unique_sl_labels)
    num_cs = len(unique_sl_labels)
    sl_mats = np.zeros((num_cs,len(data_iris),len(data_iris)))
    
    
    average = AgglomerativeClustering(n_clusters=K, linkage='average')
    average.fit_predict(dist_mat)
    av_labels = average.labels_
    av_labels = pd.DataFrame(av_labels)
    unique_av_labels = np.unique(av_labels)
    av_labels['idx'] = av_labels.index
    unique_av_labels = pd.DataFrame(unique_av_labels)
    num_ca = len(unique_av_labels)
    av_mats = np.zeros((num_ca,len(data_iris),len(data_iris)))
    
    
     
    
       
    for j in range(num_cs):
        sl_temp = sl_labels[sl_labels.iloc[:,0]==j]
        permute_sl_temp = list(itertools.product(sl_temp['idx'], sl_temp['idx']))
        for p in  permute_sl_temp:
            sl_mats[j,p[0],p[1]] = 1
    
    for j in range(num_ca):
        av_temp = av_labels[av_labels.iloc[:,0]==j]
        permute_av_temp = list(itertools.product(av_temp['idx'], av_temp['idx']))
        for p in  permute_av_temp:
            av_mats[j,p[0],p[1]] = 1
    
    cluster_mats = np.concatenate((sl_mats, av_mats), axis=0)   
    cluster_mats = cluster_mats *  Co_assoc
    cluster_list = []
    for i in range(cluster_mats.shape[0]):
        cluster_temp = cluster_mats[i]
        cluster_temp = cluster_temp[cluster_temp!=0]
        cluster_stab = np.mean(cluster_temp)
        
        if cluster_stab < 0.9:
            
            cluster_list.append(i)
            
       
    cluster_list.reverse() 
    for i in    cluster_list:
        cluster_mats = list(cluster_mats)
        cluster_mats.pop(i)
    cluster_mats = np.array(cluster_mats)  
    C = np.max(cluster_mats, axis=0)
    Cs.append(C)   
    
import seaborn as sns; 

ax = sns.heatmap(Co_assoc)





#test
  

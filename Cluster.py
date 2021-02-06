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
# m is experiment number.
m=100
data_len = int(len(data_iris))

# set alogrithms and parameters
parameter_list=[3, 5, 10, 12, 15]
algorithm_list = ['knn', 'sl']
# I permute parameters and algorithm types as a list then loop them for MultiEAC
permute_parameter_list = list(itertools.product(algorithm_list, parameter_list))


Cs = []
cooDict=[]
idx = np.zeros(size)


#Multi-EAC
for i in range(len(permute_parameter_list)):#algorithms
    k = permute_parameter_list[i][1]
    Co_assoc = np.zeros((len(data_iris),len(data_iris)))
    km = KMeans(n_clusters=k, init='random',n_init=10, max_iter=300,random_state=42)
    sl = AgglomerativeClustering(n_clusters=k, linkage='average')
    counter = np.zeros((data_len,data_len))
    Alg_i = permute_parameter_list[i][0]

    if(Alg_i == 'knn'):
        for i in range(m):# m data partitions
            data1 = sub_sampling(data_iris, size)  # shuffle and subsample %90
            data1X = data1.iloc[:,0:4]     # exculde label
            idx = data1X.index
            km_temp =km.fit_predict(data1X)    #k means predict        
            km_temp = pd.DataFrame(km_temp)
            km_temp['idx'] = idx # keep index for adding Co-Assoc
            km_temp = km_temp.set_index('idx')
            km_temp1 = np.array(km_temp)
            km_temp1 = km_temp1.reshape(1,n)
            km_temp1 = np.repeat(km_temp1,n,axis=0)
            temp_co = ((km_temp1-km_temp1.T)==0)  # To control if the elements seen in same cluster.
            for i_counter in range(len(temp_co)):
                for j_counter in range(len(temp_co)):
                    counter[idx[i_counter],idx[j_counter]]+= 1  # counter keeps the number of pairs seen in the same conter
            for tempx in range(len(temp_co)):
                for tempy in range(len(temp_co)):
                     Co_assoc[idx[tempx],idx[tempy]] += temp_co[tempx,tempy]
     
    if(Alg_i == 'sl'):            
        for i in range(m):# m data partitions
            data1 = sub_sampling(data_iris, size)  # shuffle and subsample %90
            data1X = data1.iloc[:,0:4]     # exculde label
            idx = data1X.index
            km_temp =sl.fit_predict(data1X)    #k means predict        
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
             
    Co_assoc /=  counter  # We divide Co-Assoc to counter.          
    dist_mat = Co_assoc.copy() 
    dist_mat = pd.DataFrame(dist_mat)
    dist_mat = 1 - dist_mat  
    # TODO : distance threshold? It must be life time criteria
    Z1 = linkage(Co_assoc, 'single')
    lt = pd.DataFrame(Z1)
    lt = lt.iloc[:,2:3]
    lt['dif']=lt.shift(periods=1, fill_value=0)
    diff = np.array(lt.iloc[:,0:1])-np.array(lt['dif']).reshape(len(lt['dif']),1)
    id = np.argmax(diff)
    threshold_sl = np.mean(lt.iloc[id])
    single = AgglomerativeClustering(n_clusters=None,distance_threshold=threshold_sl, linkage='single')
    single.fit_predict(dist_mat)
    
    sl_labels = single.labels_
    sl_labels = pd.DataFrame(sl_labels)
    unique_sl_labels = np.unique(sl_labels)
    sl_labels['idx'] = sl_labels.index
    unique_sl_labels = pd.DataFrame(unique_sl_labels)
    num_cs = len(unique_sl_labels) # number of cluster in SL(P_A)
    sl_mats = np.zeros((num_cs,len(data_iris),len(data_iris))) # tensor: number of cluster * n * n
    
    Z2 = linkage(Co_assoc, 'average')
    lt1 = pd.DataFrame(Z2)
    lt1 = lt1.iloc[:,2:3]
    lt1['dif']=lt1.shift(periods=1, fill_value=0)
    diff = np.array(lt1.iloc[:,0:1])-np.array(lt1['dif']).reshape(len(lt1['dif']),1)
    id = np.argmax(diff)
    threshold_al = np.mean(lt1.iloc[id])
    average = AgglomerativeClustering(n_clusters=None,distance_threshold=threshold_al,linkage='average')
    average.fit_predict(dist_mat)
    av_labels = average.labels_
    av_labels = pd.DataFrame(av_labels)
    unique_av_labels = np.unique(av_labels)
    av_labels['idx'] = av_labels.index
    unique_av_labels = pd.DataFrame(unique_av_labels)
    num_ca = len(unique_av_labels) # number of cluster in AL (P_B)
    av_mats = np.zeros((num_ca,len(data_iris),len(data_iris)))# tensor: number of cluster * n * n
    
    
     
    
    # calculating stabilities of clusters.   
    for j in range(num_cs):
        sl_temp = sl_labels[sl_labels.iloc[:,0]==j]
        permute_sl_temp = list(itertools.product(sl_temp['idx'], sl_temp['idx'])) # we create a list of pairs within clusters
        for p in  permute_sl_temp:
            sl_mats[j,p[0],p[1]] = 1 # We set pairs 1 in n*n matris and the others are 0
    
    for j in range(num_ca):
        av_temp = av_labels[av_labels.iloc[:,0]==j]
        permute_av_temp = list(itertools.product(av_temp['idx'], av_temp['idx']))
        for p in  permute_av_temp:
            av_mats[j,p[0],p[1]] = 1
    
    cluster_mats = np.concatenate((sl_mats, av_mats), axis=0) # concatenating AL and SL results into 3D tensor.   
    cluster_mats = cluster_mats *  Co_assoc  # using cluster mats as a mask for Co-assoc.
    cluster_list = []
    for i in range(cluster_mats.shape[0]):
        cluster_temp = cluster_mats[i]
        cluster_temp = cluster_temp[cluster_temp!=0]
        cluster_stab = np.mean(cluster_temp)
        
        if cluster_stab < 0.9:  # threshold clusters. 
            
            cluster_list.append(i)
            
       
    cluster_list.reverse() 
    for i in    cluster_list:
        cluster_mats = list(cluster_mats)
        cluster_mats.pop(i)
    if(len(cluster_mats)>0):
        cluster_mats = np.array(cluster_mats)  
        C = np.mean(cluster_mats, axis=0) # this max is used for combining output matricies of one algorithm.
        Cs.append(C)   
aa = np.array(Cs) 
cm = np.max(aa, axis=0)  # and the second max is used for combining the C^i for creating C_M
import seaborn as sns; 
cm1 = 1-cm

Z3 = linkage(cm1, 'average')
lt2 = pd.DataFrame(Z3)
lt2 = lt2.iloc[:,2:3]
lt2['dif']=lt2.shift(periods=1, fill_value=0)
diff = np.array(lt2.iloc[:,0:1])-np.array(lt2['dif']).reshape(len(lt2['dif']),1)
id = np.argmax(diff)
threshold_al = np.mean(lt2.iloc[id])
average = AgglomerativeClustering(n_clusters=None,distance_threshold=threshold_al,linkage='average')
average.fit_predict(cm1)
cm_labels = average.labels_
ax = sns.heatmap(cm)


'''
Z1 = linkage(Co_assoc, 'single')
SL_labels = Z1.labels_
Z2 = linkage(Co_assoc, 'average')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z1)
dn2 = dendrogram(Z2)
plt.show()
L1 = leaves_list(Z1)
L2 = leaves_list(Z2)
'''

#test
  

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
import argparse
from utils import *
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.utils.random import sample_without_replacement
from scipy.cluster.hierarchy import dendrogram, linkage,leaves_list
from matplotlib import pyplot as plt
import itertools


def run_kmeans(km, data, pc, m, n):

    co_assoc = np.zeros((len(data), len(data)))
    counter = np.zeros((len(data), len(data)))

    for i in range(m):# m data partitions
        data1 = sub_sampling(data, pc)  # shuffle and subsample %90
        data1X = data1.iloc[:, :-1]     # exculde label
        idx = data1X.index
        km_temp = km.fit_predict(data1X)    #k means predict        
        km_temp = pd.DataFrame(km_temp)
        km_temp['idx'] = idx # keep index for adding Co-Assoc
        km_temp = km_temp.set_index('idx')
        km_temp = np.array(km_temp)
        km_temp = km_temp.reshape(1, n)
        km_temp = np.repeat(km_temp, n, axis=0)
        temp_co = ((km_temp-km_temp.T)==0)  # To control if the elements seen in same cluster.

        for i_counter in range(len(temp_co)):
            for j_counter in range(len(temp_co)):
                counter[idx[i_counter],idx[j_counter]] += 1  # counter keeps the number of pairs seen in the same conter

        for tempx in range(len(temp_co)):
            for tempy in range(len(temp_co)):
                 co_assoc[idx[tempx],idx[tempy]] += temp_co[tempx,tempy]

    return co_assoc, counter

def run_sl(sl, data, pc, m, n):

    co_assoc = np.zeros((len(data), len(data)))
    counter = np.zeros((len(data), len(data)))

    for i in range(m):# m retries
        data1 = sub_sampling(data, pc)  # shuffle and subsample %90
        data1X = data1.iloc[:, :-1]     # exculde label
        idx = data1X.index
        km_temp = sl.fit_predict(data1X)    #k means predict        
        km_temp = pd.DataFrame(km_temp)
        km_temp['idx'] = idx
        km_temp = km_temp.set_index('idx')
        km_temp1 = np.array(km_temp)
        km_temp1 = km_temp1.reshape(1, n)
        km_temp1 = np.repeat(km_temp1, n, axis=0)
        temp_co = ((km_temp1 - km_temp1.T)==0)

        counter 
        for i_counter in range(len(temp_co)):
            for j_counter in range(len(temp_co)):
                counter[idx[i_counter],idx[j_counter]] += 1

        for tempx in range(len(temp_co)):
            for tempy in range(len(temp_co)):
                 co_assoc[idx[tempx],idx[tempy]] += temp_co[tempx,tempy]   
 
    return co_assoc, counter

def main(args):
    data = get_dataset(args.dataset_name)

    # set algorithms and parameters
    kmeans_sl_params_list = args.kmeans_sl_params
    algorithm_list = args.methods
    n = int(len(data) * args.pc)
    m = args.retries

    permuted_parameter_list = list(itertools.product(algorithm_list,
                                                     kmeans_sl_params_list))

    Cs = []
    cooDict=[]
    #idx = np.zeros(size)
    #Multi-EAC
    for i in range(len(permuted_parameter_list)):
        k = permuted_parameter_list[i][1]
        #Co_assoc = np.zeros((len(data),len(data)))
        km = KMeans(n_clusters=k, init='random',n_init=10, max_iter=300)
        sl = AgglomerativeClustering(n_clusters=k, linkage='single')
        #counter = np.zeros((len(data),len(data)))
        Alg_i = permuted_parameter_list[i][0]

        if Alg_i == 'kmeans':
            Co_assoc, counter = run_kmeans(km, data, args.pc, m, n)
        elif Alg_i == 'sl':
            Co_assoc, counter = run_sl(sl, data, args.pc, m, n)

        Co_assoc /=  counter  # We divide Co-Assoc to counter.          
        dist_mat = Co_assoc.copy() 
        dist_mat = pd.DataFrame(dist_mat)
        dist_mat = 1 - dist_mat  
        # TODO : distance threshold? It must be life time criteria
        single = AgglomerativeClustering(n_clusters=None,distance_threshold=3.5, linkage='single')
        single.fit_predict(dist_mat)
        
        sl_labels = single.labels_
        sl_labels = pd.DataFrame(sl_labels)
        unique_sl_labels = np.unique(sl_labels)
        sl_labels['idx'] = sl_labels.index
        unique_sl_labels = pd.DataFrame(unique_sl_labels)
        num_cs = len(unique_sl_labels)
        # tensor: number of cluster * n * n
        sl_mats = np.zeros((num_cs, len(data), len(data))) 
        
        average = AgglomerativeClustering(n_clusters=None,distance_threshold=3.5,linkage='average')
        average.fit_predict(dist_mat)
        av_labels = average.labels_
        av_labels = pd.DataFrame(av_labels)
        unique_av_labels = np.unique(av_labels)
        av_labels['idx'] = av_labels.index
        unique_av_labels = pd.DataFrame(unique_av_labels)
        num_ca = len(unique_av_labels)
        # tensor: number of cluster * n * n
        av_mats = np.zeros((num_ca, len(data), len(data)))

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
            
            if cluster_stab < 0.95:  # threshold clusters. 
                
                cluster_list.append(i)
                
           
        cluster_list.reverse() 
        for i in cluster_list:
            cluster_mats = list(cluster_mats)
            if len(cluster_mats) > 1:
                cluster_mats.pop(i)
        cluster_mats = np.array(cluster_mats)  
        C = np.max(cluster_mats, axis=0) # this max is used for combining output matricies of one algorithm.
        Cs.append(C)

    aa = np.array(Cs) 
    cm = np.max(aa, axis=0)  # and the second max is used for combining the C^i for creating C_M
    import seaborn as sns; 
    cm = 1-cm
    average = AgglomerativeClustering(n_clusters=3,linkage='single')
    average.fit_predict(cm)
    cm_labels = average.labels_
    ax = sns.heatmap(cm)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering")
    parser.add_argument('dataset_name', metavar='D', type=str,
                        help="Enter the name of the dataset")
    parser.add_argument('--methods', metavar='M', type=str, nargs='+',
                        default=['kmeans', 'sl'],
                        help="Enter the methods to be tested")
    parser.add_argument('--kmeans_sl_params', metavar='KNNSL', type=int,
                        nargs='+', default=[3, 5, 10, 12, 15],
                        help="k-means and SL parameters (k)")
    parser.add_argument('--pc', metavar='P', type=float, default=0.9,
                        help="Subsample Percentage b/w 0 and 1")
    parser.add_argument('--retries', metavar='R', type=int, default=100,
                        help="Number of retries for each experiments")
    args = parser.parse_args()
    main(args)

"""
#load iris data
data_iris=pd.read_csv('data/iris.data')
#data_cancer=pd.read_csv('data/breast-cancer-wisconsin.data')
#data_yeast=pd.read_csv('data/yeast.data')

le = preprocessing.LabelEncoder()
data_iris['Iris-setosa'] = le.fit_transform(data_iris['Iris-setosa'])


# set size of subsample
size=90
n = int(len(data_iris) * size/100)
# m is experiment number.
m=100


# set alogrithms and parameters
parameter_list=[3, 5, 10, 12, 15]
#algorithm_list = ['knn','sl']
algorithm_list = ['knn']
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
    sl = AgglomerativeClustering(n_clusters=k, linkage='single')
    counter = np.zeros((149,149))
    Alg_i = permute_parameter_list[i][0]

    if(Alg_i == 'knn'):
        for i in range(m):# m data partitions
            data1 = sub_sampling(data_iris, size/100)  # shuffle and subsample %90
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
    single = AgglomerativeClustering(n_clusters=None,distance_threshold=3.5, linkage='single')
    single.fit_predict(dist_mat)
    
    sl_labels = single.labels_
    sl_labels = pd.DataFrame(sl_labels)
    unique_sl_labels = np.unique(sl_labels)
    sl_labels['idx'] = sl_labels.index
    unique_sl_labels = pd.DataFrame(unique_sl_labels)
    num_cs = len(unique_sl_labels)
    sl_mats = np.zeros((num_cs,len(data_iris),len(data_iris))) # tensor: number of cluster * n * n
    
    
    average = AgglomerativeClustering(n_clusters=None,distance_threshold=3.5,linkage='average')
    average.fit_predict(dist_mat)
    av_labels = average.labels_
    av_labels = pd.DataFrame(av_labels)
    unique_av_labels = np.unique(av_labels)
    av_labels['idx'] = av_labels.index
    unique_av_labels = pd.DataFrame(unique_av_labels)
    num_ca = len(unique_av_labels)
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
        
        if cluster_stab < 0.95:  # threshold clusters. 
            
            cluster_list.append(i)
            
       
    cluster_list.reverse() 
    for i in    cluster_list:
        cluster_mats = list(cluster_mats)
        if len(cluster_mats) > 1:
            cluster_mats.pop(i)
    cluster_mats = np.array(cluster_mats)  
    C = np.max(cluster_mats, axis=0) # this max is used for combining output matricies of one algorithm.
    Cs.append(C)
"""   
aa = np.array(Cs) 
cm = np.max(aa, axis=0)  # and the second max is used for combining the C^i for creating C_M
import seaborn as sns; 
cm = 1-cm
average = AgglomerativeClustering(n_clusters=3,linkage='single')
average.fit_predict(cm)
cm_labels = average.labels_
ax = sns.heatmap(cm)

plt.show()

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
  

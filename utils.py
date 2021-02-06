# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 19:14:13 2020

@author: sg
"""

import random
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.cluster import KMeans

def sub_sampling(data, pc):
    data = data.sample(frac=1)
    max_idx = int(pc * len(data))
    data = data.iloc[0:max_idx,:]
    return data

def unique(list1): 
    x = np.array(list1) 

def unique(list1): 
    
    # intilize a null list 
    unique_list = [] 
      
    # traverse for all elements 
    for x in list1: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list

def get_dataset(name, eps=1e-6):
    if name == "iris":
        data = pd.read_csv('data/iris.data', header=None, index_col=None)
        le = preprocessing.LabelEncoder()
        data[data.columns[-1]] = le.fit_transform(data[data.columns[-1]])
        data = pd.DataFrame(data)
    elif name == "breast-cancer":
        data = pd.read_csv('data/breast-cancer-wisconsin.data', header=None, index_col=0)
    elif name == "optdigits":
        data = pd.read_csv('data/optdigits.tra', header=None, index_col=0)
        data = data[:100]
    elif name == "log_yeast":
        data = pd.read_csv('data/yeast.data', header=None, index_col=0, 
                            delimiter="\t")
        le = preprocessing.LabelEncoder()
        data[data.columns[-1]] = le.fit_transform(data[data.columns[-1]])
        data = np.array(data)
        y = data[:, -1][:, None]
        x = data[:, 1:-1]
        x = x + eps
        x = np.log(x)
        data = np.concatenate((x, y), axis=1)
        data = pd.DataFrame(data)
    elif name == "std_yeast":
        data = pd.read_csv('data/yeast.data', header=None, index_col=0, 
                            delimiter="\t")
        le = preprocessing.LabelEncoder()
        data[data.columns[-1]] = le.fit_transform(data[data.columns[-1]])
        data = np.array(data)
        y = data[:, -1][:, None]
        x = data[:, 1:-1]
        x = x + eps
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        x = (x - mean) / std
        data = np.concatenate((x, y), axis=1)
        data = pd.DataFrame(data)
    else:
        raise NotImplementedError("Name should be either one of 'iris', "\
                                   "'breast-cancer', 'optdigits', "\
                                   "'log_yeast' or 'std_yeast'")
    return data

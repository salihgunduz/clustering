# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 19:14:13 2020

@author: sg
"""

import random
import numpy as np
from sklearn.cluster import KMeans

def sub_sampling(data, size):
    data = data.sample(frac=1)
    percent = int(len(data) * size /100)
    data = data.iloc[0:percent,:]
    return data



# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 22:02:53 2021

@author: sg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from scipy.cluster import hierarchy

import sklearn

import sys

print("Python Version : ", sys.version)
print("Scikit-Learn Version : ", sklearn.__version__)



from sklearn import datasets

iris = datasets.load_iris()
X, Y = iris.data[:, [2,3]], iris.target

print("Dataset Features : ", iris.feature_names)
print("Dataset Target : ", iris.target_names)
print('Dataset Size : ', X.shape, Y.shape)


with plt.style.context("ggplot"):
    plt.scatter(X[:,0], X[:, 1], c=Y)
    plt.xlabel(iris.feature_names[2])
    plt.ylabel(iris.feature_names[3])
    plt.title("IRIS Dataset")
    

clusters = hierarchy.linkage(X, method="complete")

clusters[:10]


def plot_dendrogram(clusters):
    plt.figure(figsize=(20,6))
    dendrogram = hierarchy.dendrogram(clusters, labels=Y, orientation="top",leaf_font_size=9, leaf_rotation=360)
    plt.ylabel('Euclidean Distance');

plot_dendrogram(clusters)
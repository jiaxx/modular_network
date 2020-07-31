# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 15:25:39 2018

@author: Xiaoxuan Jia
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score

def merge_small_cluster(labels, min_size=10):
# group small clusters to neighboring large one
    new_labels=np.zeros(len(labels))
    l_tmp=0
    for i, l in enumerate(np.unique(labels)):
        idx = np.where(labels==l)[0] 
        if i==0:
            new_labels[idx]=l_tmp
        else:
            if len(idx)<min_size:
                new_labels[idx]=l_tmp
            else:
                l_tmp+=1
                new_labels[idx]=l_tmp
    return new_labels

def main():
    model = SpectralCoclustering(n_clusters=10, random_state=0).fit(X)

    fit_data = X[np.argsort(model.row_labels_),:][:, np.argsort(model.column_labels_)]
    plt.figure()
    plt.imshow(fit_data, cmap='bwr', vmax=0.000002, vmin=-0.000002)
    plt.show()

    # merge clusters with less than 10 units
    new_row = merge_small_cluster(model.row_labels_, min_size=10)
    new_column = merge_small_cluster(model.column_labels_, min_size=10)

    fit_data = X[np.argsort(new_row),:][:, np.argsort(new_column)]
    plt.figure()
    plt.imshow(fit_data, cmap='bwr', vmax=0.000002, vmin=-0.000002)
    plt.show()





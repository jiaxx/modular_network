# -*- coding: utf-8 -*-
"""
Created on Tue May 2 16:14:42 2017

@author: Xiaoxuan Jia
"""
from sklearn import preprocessing # 0 mean and std=1
from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt


def pca_basic(data, plot=False, threshold=0.90, n_components=None):
    """data is m_samples*n_features
    output Z is feature*samples
    """
    # feature reduction: 
    # 1. convert data structure to n_features*m_samples
    #size=np.shape(data)
    #original = np.reshape(data,[np.shape(data)[0],np.shape(data)[1]*np.shape(data)[2]*np.shape(data)[3]])
    original = data
    X = original.T
    # X: feature*sample

    #print('shape',str(np.shape(X)))

    # 2. mean normalization (ensure every feature has zero mean) and feature scaling (ensure every feature in the same scale)
    
    X_scaled = preprocessing.scale(X, axis=0)
    #print('shape scaled:', str(np.shape(X_scaled)))

    # 3 calculate covariance matrix of normalized data
    sigma = np.cov(X_scaled)

    # 4. calculate singular value decomposition of covariance matrix
    U, s, Vh = linalg.svd(sigma)
    if plot==True:
        #max(sigma.flatten())
        plt.figure(figsize=(10,3))
        plt.subplot(121)
        plt.imshow(sigma,clim=[-1,1], cmap='bwr')
        plt.subplot(122)
        ev=[]
        for k in range(len(s)):
            ev.append(sum(s[:k])/sum(s))
        plt.plot(ev)
        plt.plot([0, len(s)], [threshold, threshold],'k:')

    # 5. determine k for U_reduced: 90% of variance is retained
    # sum(s[:k])/sum(s)>=0.90
    if n_components==None:
        for k in range(len(s)):
            if sum(s[:k])/sum(s)>=threshold: # should try 80 to remove more noise
                #print(k)
                break
    else:
        k=n_components

    # 6. compute projected data Z
    U_reduce=U[:,:k]
    Z = np.dot(U_reduce.T,X_scaled)

    #print('size of transformed data:', str(np.shape(Z)))
    return Z, k

    
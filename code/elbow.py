# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:55:8 2017

@author: Xiaoxuan Jia
"""
###
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from sklearn import datasets
#from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA as RandomizedPCA
from matplotlib import pyplot as plt
from matplotlib import cm

class Elbow_k(object):

	def __init__(self, X_pca, n):
		"""X_pca/whitened data with sample*features.
		n upper limit of k values. """
		X_pca=np.array(X_pca)
		self.K=range(1,n)
		self.KM = [KMeans(n_clusters=k).fit(X_pca) for k in self.K]
		self.centroids = [k.cluster_centers_ for k in self.KM] # cluster centroids

		self.D_k = [cdist(X_pca, cent, 'euclidean') for cent in self.centroids]
		cIdx = [np.argmin(D,axis=1) for D in self.D_k]
		dist = [np.min(D,axis=1) for D in self.D_k]
		self.avgWithinSS = [sum(d)/X_pca.shape[0] for d in dist]

		# Total with-in sum of square
		self.wcss = [sum(d**2) for d in dist]
		self.tss = sum(pdist(X_pca)**2)/X_pca.shape[0]
		self.bss = self.tss-self.wcss

	def plot_elbow(self, kIdx):
		# elbow curve
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(self.K, self.avgWithinSS, 'b*-')
		ax.plot(self.K[kIdx], self.avgWithinSS[kIdx], marker='o', markersize=12, 
		markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
		plt.grid(True)
		plt.xlabel('Number of clusters')
		plt.ylabel('Average within-cluster sum of squares')
		plt.title('Elbow for KMeans clustering')

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(self.K, self.bss/self.tss*100, 'b*-')
		ax.plot(self.K[kIdx], self.bss[kIdx]/self.tss*100, marker='o', markersize=12, 
		markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
		plt.grid(True)
		plt.xlabel('Number of clusters')
		plt.ylabel('Percentage of variance explained')
		plt.title('Elbow for KMeans clustering')
		




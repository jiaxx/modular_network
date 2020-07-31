# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:36:39 2019

@author: Xiaoxuan Jia
"""
#------------# time to first spike

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

# ------fit normal distribution to calculate precision of latency to first spike
def fit_norm(data, plot=False, bins=25):
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    #print(mu, std)
    # Plot the histogram.
    if plot==True:
        plt.hist(data, density=True, alpha=0.6, bins=bins)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)

        #plt.show()
    return mu, std


def get_norm_fit_param(first_spike, group_by='trial'):
    # fit normal distribution to reflect the compactness of the distribution of first spike times in a population
    # width of first peak distribution
    # first_spike is two dim
    if group_by=='trial':
        # trial* neuron
        n_trial=first_spike.shape[0]
    if group_by=='neuron':
        # neuron*trial
        n_trial=first_spike.shape[1]

    P=np.nan*np.ones(n_trial)
    M=np.nan*np.ones(n_trial)
    for t in range(n_trial):
        if group_by=='trial':
            mu, std = fit_norm(first_spike[t,:][np.where((first_spike[t,:]>45) & (first_spike[t,:]<100))[0]])
        if group_by=='neuron':
            mu, std = fit_norm(first_spike[:,t][np.where((first_spike[:,t]>45) & (first_spike[:,t]<100))[0]])
        P[t]=std
        M[t]=mu

    return P, M

def plot_norm_fit(x, start_time=40, end_time=200):
    # x is the times of first spike of a population at a given trial
    # width of first peak distribution
    # first_spike is two dim
    fit_norm(x[np.where((x>45) & (x<100))[0]], bins=np.arange(start_time, end_time, 5), plot=True)
    return 'plot'



        



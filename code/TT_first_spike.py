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

def find_nearest(array, value, side="right"):
    idx = np.searchsorted(array, value, side=side)
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def find_first_spike(train, start_time=30, end_time=500):
    return find_nearest(np.where(train[start_time:end_time]==1)[0]+start_time, start_time, side="left")

def compute_first_spike(spikes, start_time=30, end_time=500):
    """spikes: neuron*trial*time"""
    first_spike=np.zeros((spikes.shape[0], spikes.shape[1]))*np.NaN
    for n in range(spikes.shape[0]):
        for t in range(spikes.shape[1]):
            train = spikes[n,t,:]
            if sum(np.where(train[start_time:end_time]==1)[0]+start_time)>0:
                first_spike[n,t]=find_first_spike(train, start_time=start_time, end_time=end_time)
    return first_spike

def compute_mean_first_spike(first_spike):
    # first_spike: n*trial
    mu=[]
    for n in np.arange(first_spike.shape[0]):
        mu.append(np.median(first_spike[n, np.where(first_spike[n,:]>0)]))
    mu=np.array(mu)
    return mu

# ------fit normal distribution
def fit_norm(data, plot=False, bins=25):
    # Fit a normal distribution to the data:
    mu, std = norm.fit(data)
    # Plot the histogram.
    if plot==True:
        plt.hist(data, density=True, alpha=0.6, bins=bins)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)

        plt.show()
    return mu, std

#-------plot example of first spike of one neuron across trials
def plot_first_spike_example(spikes_select, first_spike, plot=True):
    # TODO: check whether group_by condition is correct
    # spikes_select: neuron*trial*time
    # plot highlighted first spike examples for selected neurons
    spikes_trial_neuron = mau.reshape_matrix(spikes_select, group_by='trial')
    #spikes_trial_neuron: trial*neuron*time
    n_neuron=spikes_select.shape[0]
    first_spike_tmp = first_spike[:,6].astype(int)
    
    cmap = plt.cm.get_cmap('tab10')
    plt.figure(figsize=(8,5))
    for i in np.arange(n_neuron*6, n_neuron*7): #:
        if first_spike_tmp[i-n_neuron*6]>0:
            plt.plot(first_spike_tmp[i-n_neuron*6], i, 'x', color='red')
        tmp=np.where(spikes_trial_neuron[i,:250]==1)[0]
        c = cmap(0.1*i)
        plt.plot(tmp, i*np.ones(len(tmp)), '.', c=cmap(((i/n_neuron)+1)%10*0.1), markersize=5)
    plt.ylim([n_neuron*6-5, n_neuron*7+5])
    plt.xlim([0, 250])
    plt.xlabel('Time relative to stimulus onset (ms)', fontsize=14)
    plt.ylabel('Neurons', fontsize=14)
    plt.yticks([])
    plt.title('First spike on spike train for one trial', fontsize=16)

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



        



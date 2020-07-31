# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:36:39 2018

@author: Xiaoxuan Jia
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage.filters import gaussian_filter
import psth_utils as pu

#-----------prepare matrix
def reshape_matrix(spikes_select, group_by='neuron'):
    # group neurons for each trial, plot all trials
    # trial*neuron*time
    if group_by=='trial':
        spikes_trial_neuron = np.reshape(np.rollaxis(spikes_select, 1,0), (spikes_select.shape[0]*spikes_select.shape[1], spikes_select.shape[2]))
        return spikes_trial_neuron
    # group trials for each neuron, plot all neurons
    # neuron*trial*time
    if group_by=='neuron':
        spikes_neuron_trial = np.reshape(spikes_select, (spikes_select.shape[0]*spikes_select.shape[1], spikes_select.shape[2]))
        return spikes_neuron_trial

#-----------get PSTH
def get_psth(spikes_select, window=[0,200], PSTH_bintime=2, filter_window=5)
    unit_psth, time = pu.get_PSTH_alldim(spikes_select, PSTH_bintime=PSTH_bintime)
    return unit_psth, time

#-----------define find peaks function
def find_psth_peaks(x, prominence=1, width=20):
    # for each neuron and each trial
    peaks, properties = find_peaks(x, prominence=prominence, width=width)
    #print(properties["prominences"], properties["widths"])
    return peaks, properties

#----------find peaks for each trial of a group of neurons at each depth
def find_peaks_grouped_psth(unit_psth_tmp, window=[0,200], PSTH_bintime=2, filter_window=5, plot=False):
    # unit_psth_tmp is the subselected neurons, n_neuron*n_trial*n_time
    # return the peak (population PSTH) parameters for each trial
    assert len(np.shape(unit_psth_tmp))==3
    n_trial=unit_psth_tmp.shape[1]
    # extract peak features
    W=np.nan*np.ones(n_trial)
    H=np.nan*np.ones(n_trial)
    T=np.nan*np.ones(n_trial)
    for t in range(n_trial):
        x=gaussian_filter(unit_psth_tmp[:,t,window[0]:int(window[1]/PSTH_bintime)].mean(0)/PSTH_bintime*1000, filter_window)
        peaks, properties=find_psth_peaks(x, prominence=0.8, width=5)

        if len(properties["right_ips"]-properties["left_ips"])>0:
            idx = np.where((properties["right_ips"]-properties["left_ips"])==
                           max(properties["right_ips"]-properties["left_ips"]))[0]
            W2[t] = (properties["right_ips"]-properties["left_ips"])[idx]
            H2[t] = (x[peaks])[idx]
            T2[t] = peaks[idx]*PSTH_bintime

        if plot==True:
            plt.figure(figsize=(5,3))
            plt.plot(x)
            plt.plot(peaks, x[peaks], "x")
            plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
                       ymax = x[peaks], color = "C1")
            plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
                       xmax=properties["right_ips"], color = "C1")
            #plt.plot(spikes_select3[n,t,10*PSTH_bintime:100*PSTH_bintime]*1000)
            plt.show()

    return W, H, T




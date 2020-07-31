# -*- coding: utf-8 -*-
"""
Created on Thur Aug 1 11:03:40 2019

@author: Xiaoxuan Jia
"""

import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

import os, sys,glob, copy
import json
import matplotlib.pyplot as plt

    
#------------------Estimate layer from CSD
def get_probe_idx(probename):
    all_probes=np.array(['probeA', 'probeB', 'probeC', 'probeD', 'probeE', 'probeF'])
    return np.where(all_probes==probename)[0]

def get_index_min_matrix(a):
    return np.unravel_index(np.argmin(a, axis=None), a.shape)

def find_local_maxima(data, threshold=1, neighborhood_size=5, plot=True):
    """
    find local maxima of 2D array. threshold is critical
    """
    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    #minima = (data == min(data_min))
    minima = get_index_min_matrix(data_min)
    diff = ((data_max - data_min) > threshold)
    maxima[data_max<threshold] = 0
    #minima[data_min>(-1)*threshold] = 0
    

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    x=np.array(x).astype(int)
    y=np.array(y).astype(int)

    x_m=minima[1]
    y_m=minima[0]+2
        
    if plot:
        plt.imshow(data, aspect='auto', origin='image' ,cmap='jet')
        plt.autoscale(False)
        plt.plot(x,y, 'bo',  alpha=0.5)
        plt.plot(x_m,y_m, 'ro', alpha=0.5)
        plt.show()
    
    return x, y, x_m,y_m

def load_precompute_CSD(csd_path, mouseID, probename):
    """
    TODO
    """
    csd=np.load(csd_path+'/csd.npy')
    channel=np.load(csd_path+'/csd_channels.npy')
    window=np.load(csd_path+'/relative_window.npy')

    # load probe_info
    with open(csd_path+'/probe_info.json') as f:
        probe_info = json.load(f)
    channel_surf = int(probe_info['surface_channel'])

    csd_smooth = gaussian_filter(csd,[4, 8]) #[4,8]

    return csd_smooth, channel, channel_surf

def estimate_layer4(csd_smooth, channel, channel_wm):
    """
    # estimate middle layer based on first sink between 88:350ms
    channel_wm is the channel indicating wm between cortex and hippocampus
    """
    
    # 1. slice CSD gragh to desired region
    real_ch_id = channel[np.where(channel==channel_wm-10)[0][0]:-10]
    real_time_id = np.arange(csd_smooth.shape[1]) #+(start_time-250)/2500.

    region_sink = csd_smooth[np.where(channel==channel_wm-10)[0][0]:-10,:]

    middle_ch=0
    middle_time=0
    # 2. check if CSD==0
    if sum(region_sink.flatten())!=0:
        # find local maxima passing certain threshold
        plt.figure(figsize=(4,6))
        x, y, x_m,y_m = find_local_maxima(region_sink, threshold=max(region_sink.flatten())/2, plot=True)
        # _m are minimum source
        x=np.flipud(x)
        y=np.flipud(y)

        peak_time_idx=0
        peak_ch_idx=0
        
        select_idx=np.where((x<200) & (x>40) & (x<=x_m+50) & (y<y_m+10) )[0] #& (y<y_m[idx]+10) & (y>y_m[iidx]-10)
        if len(select_idx)>0:
            print('not empty')
            x_tmp = x[select_idx]
            y_tmp = y[select_idx]

            peak_time_idx = x_tmp[np.argmin(x_tmp)]
            peak_ch_idx = y_tmp[np.argmin(x_tmp)]

            # plot sliced region with sink only
            #plt.figure(figsize=(4,6))
            #plt.imshow(region_sink, extent=[window[250], window[750], channel_wm-10, channel[-1]-10],aspect='auto',
            #           origin='image',cmap='jet')
            #plt.colorbar()
            plt.scatter(peak_time_idx, peak_ch_idx, s=30, c='white')
        else:
            peak_time_idx=x_m
            peak_ch_idx=y_m
        
        print(probename, real_ch_id[peak_ch_idx])
        middle_ch=real_ch_id[peak_ch_idx]
        middle_time=real_time_id[peak_time_idx]
    return middle_ch, middle_time

def compute_csd_layer(csd_smooth, channels_cortex):
    real_ch_id = channels_cortex
    real_time_id = np.arange(csd_smooth.shape[1]) 

    region_sink = csd_smooth

    middle_ch=0
    middle_time=0
    # 2. check if CSD==0
    if sum(region_sink.flatten())!=0:
        # find local maxima passing certain threshold
        plt.figure(figsize=(4,6))
        x, y, x_m,y_m = find_local_maxima(region_sink, threshold=max(region_sink.flatten())/2, plot=True)
        # _m are minimum source
        x=np.flipud(x)
        y=np.flipud(y)

        peak_time_idx=0
        peak_ch_idx=0

        select_idx=np.where((x<200) & (x>40) & (x<=x_m+50) & (y<y_m+10) )[0] #& (y<y_m[idx]+10) & (y>y_m[iidx]-10)
        if len(select_idx)>0:
            print('not empty')
            x_tmp = x[select_idx]
            y_tmp = y[select_idx]

            peak_time_idx = x_tmp[np.argmin(x_tmp)]
            peak_ch_idx = y_tmp[np.argmin(x_tmp)]

            # plot sliced region with sink only
            #plt.figure(figsize=(4,6))
            #plt.imshow(region_sink, extent=[window[250], window[750], channel_wm-10, channel[-1]-10],aspect='auto',
            #           origin='image',cmap='jet')
            #plt.colorbar()
            plt.scatter(peak_time_idx, peak_ch_idx, s=30, c='white')
        else:
            peak_time_idx=x_m
            peak_ch_idx=y_m

        print(probename, real_ch_id[peak_ch_idx])
        middle_ch=real_ch_id[peak_ch_idx]
        middle_time=real_time_id[peak_time_idx]
    return middle_ch, middle_time

class CSD_layer(object):

    def __init__(self, csd_path, mouseID, probename, channel_wm):
        """
        load precomputed CSD data to estimate middle layer
        #basepath = '/Volumes/SD4.2/'
        basepath = '/Volumes/SD4/'
        Example
        csd_path=glob.glob(glob.glob(basepath+'*'+mouseID+'*')[0]+'/'+'*'+mouseID+'*'+probename+'*'+'sorted')[0]
        mouseID = 'mouse389262'
        probename = 'probeC'
        channel_wm = 60

        """
        self.csd_path = csd_path
        self.mouseID = mouseID
        self.probename = probename
        self.channel_wm = channel_wm

    def run_module(self):
        # load precomputed CSD data
        self.csd_smooth, self.channel, self.channel_surf = self.load_precompute_CSD(self.csd_path, self.mouseID, self.probename)
        
        self.middle_ch, self.middle_time = self.estimate_layer4(self.csd_smooth, self.channel, self.channel_wm)
        return self.middle_ch, self.middle_time 





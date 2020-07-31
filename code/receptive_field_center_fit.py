# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:09:39 2019

@author: Xiaoxuan Jia
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sys
sys.path.append("/Users/xiaoxuanj/work/work_allen/Ephys/code_library/ephys_code")

import fit_RF_2D
import ReceptiveFieldAnalysis
import RFmap_utils as RFmap
import rf_size as rs

from scipy.ndimage import gaussian_filter, label, center_of_mass
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.stats import ttest_ind

def IQR_outlier(dataset, iqr=3):
    #dataset = data.flatten()
    q1, q3= np.percentile(dataset,[25,75])
    lower_bound = q1 - (iqr/2. * q1)
    upper_bound = q3 + (iqr/2. * q3)
    return np.where((dataset<lower_bound) | (dataset>upper_bound))[0]

def RF_preselect(data, threshold=0.01):
    """
    Set threshold for total activity to remove very low activity units
    """
    if sum(data.flatten())>threshold and len(np.where(data.flatten()>0)[0])>3:
        return True
    else:
        return False
    
def RF_detect_outlier(data):
    """
    not effective for RF
    """
    index = IQR_outlier(data.flatten())
    print(index)
    if len(index)>0:
        print('outlier exist')
        return np.unravel_index(index, (9,9))
    else:
        print('no outlier')
    
def RF_threshold_method1(data, thresh=0.2, thresh_step=0.1, plot=False):
    """
    data: RF map 9*9
    dynamic thresholding to keep reasonable number of pixels for fitting
    """
    thresh = 0.2
    thresh_step = 0.1
    peak_mask = ReceptiveFieldAnalysis.get_peak_weighted_roi(data, max(data.ravel())*thresh)
    #if peak_mask==None:
        #print('skip')
        #continue

    #if threshold is not effective, use step-wise increase of threshold
    while len(np.where(peak_mask.flatten()!=0)[0])>20:
        #print thresh
        thresh+=thresh_step
        peak_mask = ReceptiveFieldAnalysis.get_peak_weighted_roi(data, max(data.ravel())*thresh)
        if np.nanmax(peak_mask)==None:
            thresh-=thresh_step
            peak_mask = ReceptiveFieldAnalysis.get_peak_weighted_roi(data, max(data.ravel())*thresh)
            
    if plot==True:
        plt.figure(figsize=(3,3))
        plt.imshow(peak_mask, cmap='Greys')
        plt.title('method1')
    return peak_mask

def RF_threshold_method2(data, plot=False):
    """
    Scaled RF
    """
    RF_scaled = gaussian_filter(pow(data/np.max(data),2),1)
    if plot==True:
        plt.figure(figsize=(3,3))
        plt.imshow(RF_scaled, cmap='Greys')
        plt.title('method2')
    return RF_scaled
    
def RF_threshold_method3(data, binary_std_thresh = 1.0, plot=False):
    """
    Gaussian filtered RF + std threshold
    """
    RF_filt = gaussian_filter(data,1)
    thresh = np.max(RF_filt) - np.std(RF_filt) * binary_std_thresh

    RF_thresh = np.copy(RF_filt)
    RF_thresh[RF_thresh < thresh] = 0
    RF_thresh[RF_thresh >= thresh] = 1
    
    if plot==True:
        plt.figure(figsize=(3,3))
        plt.imshow(RF_thresh, cmap='Greys')
        plt.title('method3')
    return RF_thresh

def center_estimator1(RF, plot=False):
    coord = np.unravel_index(np.argmax(RF),RF.shape)
    if plot==True:
        plt.figure(figsize=(3,3))
        plt.imshow(RF, cmap='Greys')
        plt.plot(coord[1],coord[0],'.',color='red',markersize=10)
        plt.title('estimator1')
    return coord[1],coord[0]

def center_estimator2(RF, plot=False):
    # worst performance
    labels = label(RF)
    center_loc = center_of_mass(RF, labels[0])
    if plot==True:
        plt.figure(figsize=(3,3))
        plt.imshow(RF, cmap='Greys')
        plt.plot(center_loc[1], center_loc[0],'.',color='red',markersize=10)
        plt.title('estimator2')
    return center_loc[1], center_loc[0]

def center_estimator3(RF, plot=False):
    params, success = fit_RF_2D.fitgaussian(RF)
    fit = fit_RF_2D.gaussian(*params)
    (height, x, y, width_x, width_y) = params
    RF_fit = fit(*np.indices(RF.shape))
    
    if plot==True:
        plt.figure(figsize=(3,3))
        ax = plt.subplot(1,1,1)
        ax.imshow(RF, cmap='Greys')
        #ax.colorbar()
        ax.grid(False)
        ax.contour(RF_fit, cmap=plt.cm.copper)
        ax.plot(y, x, '.',color='red',markersize=10)
        plt.title('estimator3')
    #print(y, x, width_x, width_y, height)
    return y, x, width_y, width_x, height, RF_fit
    
# conclusion after testing: a combination of method 1 and estimator 1 or 3 give rise to best result
# method 2 introduce some bias, method 3 is sensitive to noise and contain limited data power
# estimator 2 is worst compare to all

def test_plot(rf, df_probe):
    # plotting test
    # fix bug in the input rf! a 0 value is created at the peak
    plt.figure(figsize=(16,20/9*(len(df_probe)/9)))
    for idx in range(len(df_probe)):
        #if idx%10==0:
        data = rf[idx,:,:]
        plt.subplot(len(df_probe)/10+1,10,idx+1)
        plt.imshow(data, cmap='Greys')

        output = RF_preselect(data)
        if output:
            peak_mask = RF_threshold_method1(data)
            #RF_filt = gaussian_filter(peak_mask,0.5)

            # gaussian fit
            x1, y1, width_x, width_y, height, RF_fit = center_estimator3(peak_mask)
            # center of mass
            x2, y2 = center_estimator1(peak_mask)

            plt.plot(x2, y2,'.',color='red',markersize=10)
            plt.grid(False)
            plt.contour(RF_fit, 2, cmap=plt.cm.copper)
            #plt.plot(x1, y1, '.',color='green',markersize=10)
            plt.xticks([])
            plt.yticks([])

from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from scipy.stats import ttest_ind

def get_RF_properties(rf, df_all, Verbose=False):
    """
    for a given df unit list, calculate RF properties
    """
    # reset index in pandas
    df_all = df_all.reset_index().drop(['index'], axis=1)
    
    assert rf.shape[0]==len(df_all)
       
    # keep receptive field info for responsive units; arbitrary threshold
    df_rf=pd.DataFrame(columns = ['probe_id', 'unit_id', 'channel_id','ypos', 'ccf', 
                                  'rf', 'rf_exists','kurtosis','rf_size', 'rf_center_x1','rf_center_y1',
                                  'rf_width_x1','rf_width_y1','rf_fit_hight', 'rf_center_x2',
                                  'rf_center_y2', 'mean','max', 'var']) #'rf_b''psth','signal','p_val',

    if Verbose==True:
        plt.figure(figsize=(16,20/9*(len(df_all)/9)))
        
    for idx, row in df_all.iterrows():
        data = rf[idx,:,:]
        # get corresponding RF on screen
        data = np.flipud(data.T)
        
        probe_id=row['probe_id']
        unit_id=row['unit_id']
        channel_id=row['channel_id']
        #ypos=row['ypos']
        ccf=row['ccf']

        rf_exists, rf_kurtosis, center_loc, rf_size = rs.rf_metrics(data)
        var = fit_RF_2D.signal_detection(data.ravel()) # 2D signal in RF map
        
        if Verbose==True:
            plt.subplot(len(df_all)/10+1,10,idx+1)
            plt.imshow(data, cmap='Greys')
        
        output = RF_preselect(data)
        if output:
            peak_mask = RF_threshold_method1(data)
            RF_scaled = RF_threshold_method2(data)
            
            # gaussian fit on thresholded pixels
            x1, y1, width_x, width_y, height, RF_fit = center_estimator3(peak_mask)
            # center of mass on gaussian smoothed RF
            x2, y2 = center_estimator2(RF_scaled)
            
            if Verbose==True:
                #if idx%10==0:
                plt.plot(x2, y2,'.',color='red',markersize=10)
                plt.grid(False)
                plt.contour(RF_fit, 2, cmap=plt.cm.copper)
                #ax.plot(x1, y1, '.',color='green',markersize=10)
                plt.xticks([])
                plt.yticks([])
            
            df_rf = df_rf.append({'probe_id':probe_id, 
                                'unit_id':unit_id, 
                                'channel_id':channel_id, 
                                #'ypos': ypos,
                                'ccf': ccf, 
                                #'signal':snr,
                                #'p_val':p,
                                'rf':data, 
                                #'rf_b':rf_b[n],
                                #'psth':psth.mean(0), 
                                'rf_exists': rf_exists,
                                'kurtosis': rf_kurtosis,
                                'rf_size': rf_size[0]*rf_size[1],
                                'rf_center_x1': x1,
                                'rf_center_y1': y1,
                                'rf_width_x1': width_x,
                                'rf_width_y1': width_y,
                                'rf_fit_hight': height,
                                'rf_center_x2': x2,
                                'rf_center_y2': y2,
                                'mean': np.nanmean(data.flatten()),
                                'max':max(data.flatten()),
                                'var':var},
                                ignore_index=True) 
            
        #print((data-rf[idx,:,:]).flatten().sum())

    plt.tight_layout()
    #plt.savefig('/Users/xiaoxuanj/work/work_allen/Ephys/figures/RF/'+mouseID+'_RF/'+probename+'_gabor_20deg_all_psth.pdf')
    return df_rf



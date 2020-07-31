# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:44:39 2019

@author: Xiaoxuan Jia
"""
    
# plot raster with different shapes
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import find_peaks

import visualization_utils as vu

color_bank = {'probeA':'r',
             'probeB':'brown',
             'probeC': '#E3CF57',
             'probeD': 'green',
             'probeE': 'purple',
             'probeF': 'blue'}
             
color_cmaps = ['r','brown','#E3CF57','green','purple','blue']

def get_area_for_probe(probenames):
    areas_all = ['AM','PM','V1','LM','AL','RL']
    probes_all = ['probeA','probeB','probeC','probeD','probeE','probeF']
    return [areas_all[np.where(np.array(probes_all)==probe)[0][0]] for probe in probenames]


def find_psth_peaks(x, prominence=1, width=20):
    peaks, properties = find_peaks(x, prominence=prominence, width=width)
    #print(properties["prominences"], properties["widths"])
    return peaks, properties

def plot_raster_pop(matrix):
    """matrix is 2D binary array of neuron/trial by time bins"""
    #plot raster for all units of a given trial
    # matrix is neuron/trial by time
    plt.figure(figsize=(30,20*matrix.shape[0]/100.))
    for i,temp in enumerate(matrix):
        # plot raster for all units in a given time window or trial
        x = np.where(temp!=0)[0]
        y = np.ones(len(x))*(i+1)
        plt.plot(x, y, '.', c='k')
        plt.ylim([-1,len(matrix)])
        plt.tick_params(direction='out')
        plt.xticks([])
        plt.yticks([])
        #plt.xlabel('Time (sec)')
        #plt.ylabel('Units')    

def raster_sorted_time(matrix, t_duration=2000, trial_select=40, figsize=(40,4), markersize=0.8, linewidth=0.5):
    """
    # matrix is neuron*trial*time
    # t_duration: duration of each trial
    # plot raster for neuron*time(over trials)
    """
    spikes_new = matrix[:,:,:t_duration].reshape(matrix[:,:,:t_duration].shape[0], 
                                                     matrix[:,:,:t_duration].shape[1]*matrix[:,:,:t_duration].shape[2])
    n_neuron=matrix.shape[0]
    # plot
    # color different areas differently
    plt.figure(figsize=figsize)
    for i in np.arange(n_neuron): #:
        tmp=np.where(spikes_new[i,:t_duration*trial_select]==1)[0]
        plt.plot(tmp, i*np.ones(len(tmp)), '.', c='k', markersize=markersize)
    # plot separation of trials
    for i in range(trial_select):
        plt.plot([i*t_duration,i*t_duration],[0,n_neuron],':r', linewidth=linewidth)
    plt.xlim([-1000, t_duration*trial_select+1000])
    return ''

def raster_neuron_time(matrix, separations, t_duration=2000, trial_select=40, figsize=(40,4), markersize=0.8, color_cmaps=['r','brown','#E3CF57','green','purple','blue']):
    """
    # matrix is neuron*trial*time
    # t_duration: duration of each trial
    # plot raster for neuron*time(over trials)
    """
    spikes_new = matrix[:,:,:t_duration].reshape(matrix[:,:,:t_duration].shape[0], 
                                                     matrix[:,:,:t_duration].shape[1]*matrix[:,:,:t_duration].shape[2])
    n_neuron=matrix.shape[0]
    # plot
    cmap = plt.cm.get_cmap('tab10')
    # color different areas differently
    plt.figure(figsize=figsize)
    for i in np.arange(n_neuron): #:
        tmp=np.where(spikes_new[i,:t_duration*trial_select]==1)[0]
        for j in range(len(separations)-1):
            if i in np.arange(separations[j], separations[j+1]):
                plt.plot(tmp, i*np.ones(len(tmp)), '.', c=color_cmaps[j], markersize=markersize) #cmap((j+1)*0.1)
    # plot separation of trials
    for i in range(trial_select):
        plt.plot([i*t_duration,i*t_duration],[0,n_neuron],':k', linewidth=0.5)
    plt.xlim([-1000, t_duration*trial_select+1000])
    plt.axis('off')

    return spikes_new


def raster_groupby_trial(matrix, trial_select=40, figsize=(15,20), markersize=2):
    """
    # matrix is neuron*trial*time
    # plot whole population for each trial, stack trials
    # color indicates each trial
    """
    spikes_trial_neuron = np.reshape(np.rollaxis(matrix, 1,0), (matrix.shape[0]*matrix.shape[1], matrix.shape[2]))
    n_neuron=matrix.shape[0]
    # color represents each trial, plot population activity for each trial
    cmap = plt.cm.get_cmap('tab10')
    plt.figure(figsize=figsize)
    for i in np.arange(n_neuron*trial_select): #:
        tmp=np.where(spikes_trial_neuron[i,:]==1)[0]
        c = cmap(0.1*i)
        plt.plot(tmp, i*np.ones(len(tmp)), '.', c=cmap(((i/n_neuron)+1)%10*0.1), markersize=markersize)
    plt.ylim([-10, n_neuron*trial_select+10])
    #plt.savefig('/Users/xiaoxuanj/work/work_allen/Ephys/figures/visualization/mouse'+mouse_ID+'_grating_L1_cluster3_matched.png')
    return spikes_trial_neuron

def raster_movie_all_units(matrix, figsize=(15,20), markersize=0.03):
    """
    matrix: neuron*trial*time
    movie stimulus has very limited trials
    """
    assert matrix.shape[1]<matrix.shape[2]
    n_neuron=matrix.shape[0]
    n_trial=matrix.shape[1]
    # all neurons for each trial, stacked across trials
    spikes_movie_neuron_trial = np.reshape(np.rollaxis(matrix, 1,0), (matrix.shape[0]*matrix.shape[1], matrix.shape[2]))
    # plot
    # color represents each trial, plot population activity for each trial
    cmap = plt.cm.get_cmap('tab10')
    plt.figure(figsize=figsize)
    for i in np.arange(n_neuron*n_trial): #:
        tmp=np.where(spikes_movie_neuron_trial[i,:]==1)[0]
        c = cmap(0.1*i)
        plt.plot(tmp, i*np.ones(len(tmp)), '.', c=cmap(((i/n_neuron)+1)%10*0.1), markersize=markersize)
    plt.ylim([-10, n_neuron*n_trial+10])

    return spikes_movie_neuron_trial

def get_probe_dict(df):
    # for each probe
    probenames = df.probe_id.unique().astype(str)
    separations = [0]
    #separations[-1]=separations[-1]-1
    for probe in probenames:
        index = np.where(df.probe_id==probe)[0]
        separations = np.concatenate([separations, [index[-1]+1]],axis=0)

    print(separations)
    print(probenames)

    areas = vu.map_probe_to_area(probenames)
    # for each area
    sub_areas=[]
    sub_separations = [0]
    for probe in probenames:
        tmp=np.array(df[df.probe_id==probe].ccf.unique()).astype(str)
        tmp=tmp[np.where(tmp!='none')[0]]
        for ccf in tmp:
            sub_areas.append(ccf)
            index = np.where((df.probe_id==probe) & (df.ccf==ccf))[0]
            sub_separations = np.concatenate([sub_separations, [index[-1]+1]],axis=0)

    print(sub_separations)
    print(sub_areas)

    # number corresponds to how many substructures on each probe
    probe_dict={}
    probe_dict['probeA']={'range': [np.array(sub_separations*rep)[:5]], 'areas': sub_areas[:4]}
    probe_dict['probeB']={'range': [np.array(sub_separations*rep)[4:9]], 'areas': sub_areas[4:8]}
    probe_dict['probeC']={'range': [np.array(sub_separations*rep)[8:12]], 'areas': sub_areas[8:11]}
    probe_dict['probeD']={'range': [np.array(sub_separations*rep)[11:16]], 'areas': sub_areas[11:15]}
    probe_dict['probeE']={'range': [np.array(sub_separations*rep)[15:20]], 'areas': sub_areas[15:19]}
    probe_dict['probeF']={'range': [np.array(sub_separations*rep)[19:]], 'areas': sub_areas[19:]}

    return probe_dict


def raster_probe(matrix, probe_dict, savefig=False):
    # raster as a function of brain area for each probe
    # label brain region with probe_dict

    # after reshape, matrix is neuron1*20 trials . then neuron2*20 trials
    if matrix.shape[1]>matrix.shape[2]:
        spikes_movie_trial = np.reshape(np.rollaxis(matrix, 2,1), (matrix.shape[0]*smatrix.shape[2], matrix.shape[1]))
        rep = matrix.shape[2]
    else:
        spikes_movie_trial = np.reshape(matrix, (matrix.shape[0]*matrix.shape[1], matrix.shape[2]))
        rep = matrix.shape[1]

    probenames=probe_dict.keys()

    plt.figure(figsize=(20,40))
    cmap = plt.cm.get_cmap('tab10')
    for probe in probenames:
        tmp_dict = probe_dict[probe]
        print(tmp_dict['range'][0][0], tmp_dict['range'][0][-1])
        for i in np.arange(tmp_dict['range'][0][0], tmp_dict['range'][0][-1]): #:
            tmp=np.where(spikes_movie_trial[i,:40000]==1)[0]
            plt.plot(tmp, i*np.ones(len(tmp)), '.', c=cmap(((i/rep)+1)%10*0.1), markersize=0.5)

        plt.yticks(sub_separations,  tmp_dict['areas'], fontsize=40)
        if savefig==True:
            plt.savefig('/Users/xiaoxuanj/work/work_allen/Ephys/figures/visualization/mouse412792_tensor_movie1_'+probe+'.png')
        # image not shown in notebook but saved

#-----------PSTH. by area
def plot_normalized_PSTH(spikes, df):
    #plot normalized PSTH for each area
    probes = df.probe_id.unique()
    areas = df.ccf.unique()
    matrix = spikes.mean(1).mean(1).mean(1)
    # normalize to max response for each unit and then average aross 

    for probe in probes:
        plt.figure(figsize=(16, 2))
        areas = df[df.probe_id==probe].ccf.unique()
        for idx, area in enumerate(areas):
            index = np.where((df.probe_id.values==probe) & (df.ccf.values==area))[0]
            if len(index)>0:
                tmp = matrix[index, :1000]

                unit_psth, time = pu.get_PSTH(tmp, PSTH_bintime=10)

                psth_norm = np.zeros(np.shape(unit_psth))
                for i in range(unit_psth.shape[0]):
                    norm = unit_psth[i,:]/max(unit_psth[i,:])
                    if max(norm)==1:
                        psth_norm[i,:] = gaussian_filter(norm, 2)
                    else:
                        psth_norm[i,:] = np.zeros(len(norm))*np.nan

                plt.subplot(1,4,idx+1)
                plt.errorbar(time, np.nanmean(psth_norm, axis=0), np.nanstd(psth_norm, axis=0)/np.sqrt(psth_norm.shape[0]))
                plt.title(area+' '+'n='+str(len(index)), fontsize=16)
                #plt.ylim([0.2, 0.8])
                plt.xlabel('Time (sec)', fontsize=14)

                if idx==0:
                    plt.ylabel(probe, fontsize=14)
        #plt.savefig('/Users/xiaoxuanj/work/work_allen/Ephys/figures/waveforms/RS1_RS2_response_properties/PSTH_gratings_preferredori_highcon.pdf')






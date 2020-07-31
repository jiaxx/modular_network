# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 11:47:39 2018

@author: Xiaoxuan Jia
"""
# # first estimate from CSD then validate with layer PSTH

# two layer
import numpy as np
import pandas as pd

def partition_depth(depth, channel_id, layer):
    depth_sorted=np.sort(depth)
    channel_sorted=channel_id[np.argsort(depth)]
    if layer==2:
        idx = np.where(abs(depth_sorted-0.5)==min(abs(depth_sorted-0.5)))[0]
        if len(idx)>1:
            idx=idx[0]     
        return [channel_sorted[-1], channel_sorted[idx], channel_sorted[0]]
    if layer==3:
        idx1 = np.where(abs(depth_sorted-0.4)==min(abs(depth_sorted-0.4)))[0]
        if len(idx1)>1:
            idx1=idx1[0]
        idx2 = np.where(abs(depth_sorted-0.6)==min(abs(depth_sorted-0.6)))[0]
        if len(idx2)>1:
            idx2=idx2[0]
        return [channel_sorted[-1], channel_sorted[idx2], channel_sorted[idx1], channel_sorted[0]]

def partition_layer(layers, channel_id, layer):
    layers_sorted=np.sort(layers)
    channel_sorted=channel_id[np.argsort(layers)]
    if layer==2:
        idx = np.where(abs(layers_sorted-4)==min(abs(layers_sorted-4)))[0]
        if len(idx)>1:
            idx=idx[0]     
        return [min(channel_sorted), channel_sorted[idx], max(channel_sorted)]
    if layer==3:
        idx1 = np.where(abs(layers_sorted-3)==min(abs(layers_sorted-3)))[0]
        if len(idx1)>1:
            idx1=idx1[0]
        idx2 = np.where(abs(layers_sorted-5)==min(abs(layers_sorted-5)))[0]
        if len(idx2)>1:
            idx2=idx2[0]
        return [min(channel_sorted), channel_sorted[idx2], channel_sorted[idx1], max(channel_sorted)]

def get_layer_dict(df_layer, mouse_ID, layer=2, condi='layer'):
    """
    layer=2: 2 partition: layers 1:4; layers 5:6
    layer=3: 3 partition: layers 1:3; layer 4; layer 5:6
    """
    # check whether mouse exist in production
    if mouse_ID in df_layer.mouse_id.unique():
        df_tmp = df_layer[df_layer.mouse_id==mouse_ID]
        probenames = df_tmp.probe_id.unique()
        print(mouse_ID)
        #print(probenames)

        dict_layer={}
        for probe in probenames:
            areas = df_tmp[df_tmp.probe_id==probe].area.unique().astype('str')
            area = [a for a in areas if 'VIS' in a]
            if len(area)>0:
                df_ttmp = df_tmp[(df_tmp.probe_id==probe) & (df_tmp.area==area[0])]
                #print(df_ttmp.cortical_depth.unique())
                if len(df_ttmp.cortical_depth.unique())>1:
                    channel_ids = df_ttmp.channel_id.values
                    if condi=='depth':
                        depth = df_ttmp.cortical_depth.values
                        amo = partition_depth(depth, channel_ids, layer)
                        amo = list(np.array(amo).astype('int'))
                    if condi=='layer':
                        layers = df_ttmp.cortical_layer.values
                        amo = partition_layer(layers, channel_ids, layer)
                        amo = list(np.array(amo).astype('int'))
                    dict_layer[probe]=amo

                else:
                    print(probe+' no cortical depth label; all values=0')
            else:
                print(probe+' has no units in cortex')
        #print(dict_layer)
        return dict_layer
    else:
        print('mouse'+mouse_ID+' does not exist in production')

probe_areas={'probeA':'AM',
           'probeB':'PM',
           'probeC':'V1',
           'probeD':'LM',
           'probeE':'AL',
           'probeF':'RL'}

# sequence of saved CCGs
probes_all = np.array(['probeA','probeB','probeC','probeD','probeE','probeF',])

def get_labels(probenames, layer=2):
    # superficial at top
    if layer==2:
        labels=[]
        label_idx=[]
        for probe in probenames:
            labels.append(probe_areas[probe]+'-d')
            labels.append(probe_areas[probe]+'-s')
            label_idx.append(np.where(probes_all==probe)[0]*2+1)
            label_idx.append(np.where(probes_all==probe)[0]*2)
            
    if layer==3:
        labels=[]
        label_idx=[]
        for probe in probenames:
            labels.append(probe_areas[probe]+'-d')
            labels.append(probe_areas[probe]+'-m')
            labels.append(probe_areas[probe]+'-s')
            label_idx.append(np.where(probes_all==probe)[0]*3+2)
            label_idx.append(np.where(probes_all==probe)[0]*3+1)
            label_idx.append(np.where(probes_all==probe)[0]*3)
    
    return labels, label_idx

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 3 15:16:39 2019

@author: Xiaoxuan Jia
"""

# prepare matrix: condense units according to channels) 
# plot graph as a function of depth (channel_id)
# generate network plot within area and between areas


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os, sys, glob
import networkx as nx
import matplotlib.pyplot as plt
plt.style.use('default')

sys.path.append("/Users/xiaoxuanj/work/work_allen/Ephys/code_library/ephys_code/")
import draw_neural_network as NN


color_bank = {'probeA':'r',
                 'probeB':'brown',
                 'probeC': '#ff8c00',
                 'probeD': 'green',
                 'probeE': 'purple',
                 'probeF': 'blue'}

def plot_circle(t, r=1, center=[0,0]):
    x = r*np.cos(t) + center[0]
    y = r*np.sin(t) + center[1]
    return x, y

# set nodes location according to depth (channel_id)
def set_nodes_position(df, depth_all, new_probes, plot=False):  
    """
    set node location as function of channel id (depth)
    """
    n_nodes = len(depth_all)
    pos_probe={}
    pos_probe['probeC']=np.arange(250,310,0.6)/180.*np.pi
    pos_probe['probeD']=np.arange(180,240,0.6)/180.*np.pi
    pos_probe['probeE']=np.arange(120,180,0.6)/180.*np.pi
    pos_probe['probeF']=np.arange(60,120,0.6)/180.*np.pi
    pos_probe['probeA']=np.arange(0,60,0.6)/180.*np.pi
    pos_probe['probeB']=np.arange(310,360,0.4)/180.*np.pi


    probenames = df.probe_id.unique()
    t=[]
    for probe in probenames:
        chs = df[df.probe_id==probe].channel_id.unique()
        chs = chs-min(chs)
        if len(pos_probe[probe])>max(chs):
            t.append(pos_probe[probe][chs+1])
        else:
            print('Not enough nodes in '+probe+'. Add nodes!')
    t=np.concatenate(t, axis=0)
    print(len(t))

    pos={}
    for idx, a in enumerate(t):
        x, y = plot_circle(a)
        pos[idx]=np.array([x, y])

    if plot==True:
        FG = nx.Graph()
        # draw nodes
        for p in probenames:
            nx.draw_networkx_nodes(FG,pos,
                                   nodelist=list(probe_list[p]),
                                   node_color=color_bank[p],
                                   node_size=10,
                                   alpha=0.6)
        plt.axis('equal')

    # nodes id list for each probe
    probe_list={}
    for idx, probe in enumerate(probenames):
        probe_list[probe]=np.arange(n_nodes)[np.where(new_probes==probe)[0]]

    labels={}
    for i in range(n_nodes):
        labels[i]=int(depth_all[i])
    
    return pos, probe_list, labels

# get unique depth for each probe
def get_unique_depth_matrix(df, probenames):
    depth={}
    depth_all=[]
    new_probes=[]
    for probe in probenames:
        tmp = df[df.probe_id==probe].channel_id.unique()
        depth[probe]=list(tmp)
        depth_all.append(list(tmp))
        new_probes.append([probe]*len(tmp))
    depth_all = np.concatenate(depth_all, axis=0)
    new_probes = np.concatenate(new_probes, axis=0)
    n_nodes = len(depth_all)
    return depth_all, new_probes, n_nodes

# create matrix according to channel
def condense_channel_matrix(tmp_X, df, depth_all, new_probes, case='between'):
    # combine units for each channel 
    new_matrix=np.zeros((len(depth_all), len(depth_all)))*np.NaN
    for i in range(len(depth_all)):
        for j in range(len(depth_all)):
            tmp_i = np.where((df.probe_id==new_probes[i]) & (df.channel_id==depth_all[i]))[0]
            tmp_j = np.where((df.probe_id==new_probes[j]) & (df.channel_id==depth_all[j]))[0]
            # removed within area
            if case=='between':
                if new_probes[i]!=new_probes[j]: 
                    if len(tmp_i)>1 or len(tmp_j)>1:
                        amo = np.nanmean(tmp_X[np.ix_(tmp_i, tmp_j)].flatten())
                    else:
                        amo = tmp_X[tmp_i, tmp_j]
                    new_matrix[i,j]=amo
            elif case=='within':
                if new_probes[i]==new_probes[j]: 
                    if len(tmp_i)>1 or len(tmp_j)>1:
                        amo = np.nanmean(tmp_X[np.ix_(tmp_i, tmp_j)].flatten())
                    else:
                        amo = tmp_X[tmp_i, tmp_j]
                    new_matrix[i,j]=amo
            elif case=='all':
                if len(tmp_i)>1 or len(tmp_j)>1:
                    amo = np.nanmean(tmp_X[np.ix_(tmp_i, tmp_j)].flatten())
                else:
                    amo = tmp_X[tmp_i, tmp_j]
                new_matrix[i,j]=amo
            else:
                print('case not specified!')
    return new_matrix


# plot circle graph relative to source
def plot_nx_graph_source(edgeP, edgeN, source_probe, pos, probe_list, labels, n_nodes, probenames, threshold=0.000002):
    """
    edges matrix with weights connecting nodes
    probe: source probe
    pos, probe_list, labels: position of nodes
    
    """
    print('source area: '+source_probe)
    
    FG = nx.Graph()
    # draw nodes
    for probe in probenames:
        nx.draw_networkx_nodes(FG,pos,
                               nodelist=list(probe_list[probe]),
                               node_color=color_bank[probe],
                               node_size=10,
                               alpha=0.6)

    # add nodes labels  
    #nx.draw_networkx_labels(FG, pos, labels,font_size=8)
    plt.axis('equal')

    # add positive edge
    for i in probe_list[source_probe]:
        for j in range(n_nodes):
            if i!=j:
                if abs(edgeP[i,j])>threshold:
                    FG.add_weighted_edges_from([(i,j, edgeP[i,j])])

    elarge=[(u,v) for (u,v,d) in FG.edges(data=True) if d['weight'] >threshold]
    print(len(elarge))
    nx.draw_networkx_edges(FG,pos,edgelist=elarge,width=1,edge_color='r', alpha=0.3)

    # add negative edge
    for i in probe_list[source_probe]:
        for j in range(n_nodes):
            if i!=j:
                if abs(edgeN[i,j])>threshold:
                    FG.add_weighted_edges_from([(j,i, edgeN[i,j])])
    elarge=[(u,v) for (u,v,d) in FG.edges(data=True) if d['weight'] <-threshold]
    print(len(elarge))
    nx.draw_networkx_edges(FG,pos,edgelist=elarge,width=1,edge_color='b', alpha=0.3)
    plt.axis('off')

def plot_within_area_network(depth_all, edges22, edges33, source_probe='probeC'):
    """
    within area network plotted as two parallele layers
    """
    probenames_reordered = ['probeC', 'probeD', 'probeE', 'probeF', 'probeB', 'probeA']
    df_amo = pd.DataFrame()
    df_amo['depth']=depth_all
    df_amo['probe_id']=np.concatenate([[probe]*len(depth[probe]) for probe in probenames_reordered], axis=0)

    M1=edges22
    M2=edges33
    conn1=[]
    conn2=[]
    count=0
    meta = {}
    D = {}
    for idx1, probe1 in enumerate(probenames_reordered):
        for idx2, probe2 in enumerate(probenames_reordered):
            if probe1==probe2:
                meta[probe1]=count
                d = ((depth[probe1][-1]-depth[probe1][0])/2-1)*20
                D[probe1]=d
                print(probe1, probe2, count, d)

            conn1.append(M1[np.ix_(np.where(df_amo.probe_id==probe1)[0], np.where(df_amo.probe_id==probe2)[0])])
            conn2.append(M2[np.ix_(np.where(df_amo.probe_id==probe1)[0], np.where(df_amo.probe_id==probe2)[0])])
            count+=1

    # from left to right
    print(D[source_probe])
    idx = meta[source_probe]
    tmp = conn1[idx]*2000000
    tmp[tmp>2]=2
    tmp[tmp<-2]=-2
    weights1 = [tmp]
    tmp = conn2[idx]*2000000
    tmp[tmp>2]=2
    tmp[tmp<-2]=-2
    weights2 = [tmp]

    ## plot1: plot one weight matrix
    #fig = plt.figure(figsize=(6, 6))
    #ax = fig.gca()
    #ax.axis('off')
    #NN.draw_neural_net(ax, .1, .6, .1, .9, [weights1[0].shape[0], weights1[0].shape[1],  ], weights1)

    ## plot2: plot both directions of weight matrix
    #fig = plt.figure(figsize=(6, 6))
    #ax = fig.gca()
    #ax.axis('off')
    #NN.draw_neural_netbi(ax, .1, .6, .1, .9, [weights1[0].shape[0], weights1[0].shape[1],  ], weights1, weights2)

    ## plot3: nodes organized by channel number
    ## all channels with responsive neurons in cortex
    left, right, bottom, top = .1, .6, .1, .9
    chs = depth[source_probe]-depth[source_probe][0]
    layer_sizes = [max(chs), max(chs), ]
    CH = [chs]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.axis('off')
    NN.draw_neural_netbich(ax, .1, .6, .1, .9, CH, layer_sizes, weights1, weights2, plotw2=True)
    #fig.savefig('/Users/xiaoxuanj/work/work_allen/Ephys/figures/func_connect/visualize_network/mouse'+mouse_ID+'_subnetwork23_v1_AM.png')
    plt.gca().invert_yaxis()
        


def plot_between_area_matrix(input_X, df, probenames):
    # deep at top
    plt.figure(figsize=(15,15))
    for idx1, probe1 in enumerate(probenames):
        for idx2, probe2 in enumerate(probenames):
            tmp_X = input_X[np.where(df.probe_id==probe1)[0],:][:, np.where(df.probe_id==probe2)[0]]
            plt.subplot(len(probenames), len(probenames), idx1*len(probenames)+idx2+1)
            plt.imshow(tmp_X, vmax=0.000001,vmin=-0.000001,cmap='bwr')
            plt.ylabel(probe1)
            plt.xlabel(probe2)
    #plt.colorbar()
    plt.tight_layout()


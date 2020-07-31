# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:34:46 2016

@author: Xiaoxuan Jia
"""

import numpy as np
from scipy import stats
from scipy.ndimage.filters import gaussian_filter


def CSD(lfp):
    """Calculate CSD based on Stoelzel et al. (Swadlow) 2009.
    positive is sink and negative is source."""
    
    # duplicate first and last depth as per Stoelzel et al. (Swadlow) 2009
    
    lfp.append(lfp[-1])
    lfpp=[lfp[0]]
    lfpp.extend(lfp)

    # smooth across depth
    lfp = np.array(lfpp)
    lfp_smooth=np.zeros([np.shape(lfp)[0]-2, np.shape(lfp)[1]])
    for t in range(np.shape(lfp)[1]): #time
        for d in range(1,np.shape(lfp)[0]-1):  #depth
            lfp_smooth[d-1,t]=0.25*(lfp[d-1,t]+2*lfp[d,t]+lfp[d+1,t])

    # csd
    csd = np.zeros([np.shape(lfp_smooth)[0]-2, np.shape(lfp)[1]])
    for t in range(np.shape(lfp)[1]): #time
        for d in np.arange(1, np.shape(lfp_smooth)[0]-1):
            csd[d-1,t]=(1/0.04)*(lfp_smooth[d+1,t]-2*lfp_smooth[d,t]+lfp_smooth[d-1,t])


    aa=np.array([[0.1, 0.3, 0.1],[0.3, 0.5, 0.3],[0.1, 0.3, 0.1]])   #2D smoothing filter
    aa=aa/sum(aa)

    csd_smooth=conv2(csd,aa,'same')
    
    plt.figure()
    ax = sns.heatmap(np.flipud(csd[:,:1001]))
    ax.set(xticks=np.arange(0,1001,250))
    ax.set(xticklabels=np.arange(0,1001,250)/2500.)
    #ax.set(yticks=np.arange(0,1001,250))
    #ax.set(xticklabels=np.arange(0,1001,250)/2500.)
    return(csd, csd_smooth)

def CSD_plot(csd, sigma):
    """Plot 2D smoothed CSD."""

    csd_2Dsmooth = gaussian_filter(csd_smooth[:,:751]/1000.0,sigma)

    plt.figure()
    ax = plt.imshow(np.flipud(csd_2Dsmooth), cmap='jet',aspect=0.005, extent=[0,0.3,56,148]) # top close to surface
    #ax.set(xticks=np.arange(0,1001,250))
    #ax.set(xticklabels=np.arange(0,1001,250)/2500.)
    plt.title('CSD')
    plt.colorbar()
    plt.grid('off')
    plt.axis('on')
    plt.xlabel('Time')

    
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:41:03 2019

@author: Xiaoxuan Jia
"""
###use GPU for FFT since it's all about matrix multiplication

import numpy as np
from scipy import stats
import scipy

def xcorrfft(a,b,NFFT):
    # first dimention of a should be length of time
    CCG = np.fft.fftshift(np.fft.ifft(np.multiply(np.fft.fft(a,NFFT), np.conj(np.fft.fft(b,NFFT)))))
    return CCG

def nextpow2(n):
    """get the next power of 2 that's greater than n"""
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return 2**m_i


def jitter(data, l):
    # matlab reshape default is Fortran order, but python default is 'C' order, when converting code, need to specify order
    """Jitter correction adapted from Amin. Used in several neuron papers.
       Jittering multidemntational logical data where 
       0 means no spikes in that time bin and 1 indicates 
       a spike in that time bin.
       First dimention should be time and second should be the trial number.
       data: time*trial*ori
       l: time window
    """
    if len(np.shape(data))>3:
        flag = 1
        sd = np.shape(data)
        data = np.reshape(data,(np.shape(data)[0],np.shape(data)[1],len(data.flatten())/(np.shape(data)[0]*np.shape(data)[1])), order='F')
    else:
        flag = 0

    psth = np.mean(data,axis=1)
    length = np.shape(data)[0]

    if np.mod(np.shape(data)[0],l):
        data[length:(length+np.mod(-np.shape(data)[0],l)),:,:] = 0
        psth[length:(length+np.mod(-np.shape(data)[0],l)),:]   = 0

    if np.shape(psth)[1]>1:
        dataj = np.squeeze(np.sum(np.reshape(data,[l,np.shape(data)[0]//l,np.shape(data)[1],np.shape(data)[2]], order='F'), axis=0))
        psthj = np.squeeze(np.sum(np.reshape(psth,[l,np.shape(psth)[0]//l,np.shape(psth)[1]], order='F'), axis=0))
    else:
        dataj = np.squeeze(np.sum(np.reshape(data,l,np.shape(data)[0]//l,np.shape(data)[1], order='F')))
        psthj = np.sum(np.reshape(psth,l,np.shape(psth)[0]//l, order='F'))


    if np.shape(data)[0] == l:
        dataj = np.reshape(dataj,[1,np.shape(dataj)[0],np.shape(dataj)[1]], order='F');
        psthj = np.reshape(psthj,[1,np.shape(psthj[0])], order='F');

    psthj = np.reshape(psthj,[np.shape(psthj)[0],1,np.shape(psthj)[1]], order='F')
    psthj[psthj==0] = 10e-10

    corr = dataj/np.tile(psthj,[1, np.shape(dataj)[1], 1]);
    corr = np.reshape(corr,[1,np.shape(corr)[0],np.shape(corr)[1],np.shape(corr)[2]], order='F')
    corr = np.tile(corr,[l, 1, 1, 1])
    corr = np.reshape(corr,[np.shape(corr)[0]*np.shape(corr)[1],np.shape(corr)[2],np.shape(corr)[3]], order='F');

    psth = np.reshape(psth,[np.shape(psth)[0],1,np.shape(psth)[1]], order='F');
    output = np.tile(psth,[1, np.shape(corr)[1], 1])*corr

    output = output[:length,:,:]
    return output

def jitter_1d(data, l):
    # matlab reshape default is Fortran order, but python default is 'C' order, when converting code, need to specify order
    """Jitter correction adapted from Amin. Used in several neuron papers.
       Jittering multidemntational logical data where 
       0 means no spikes in that time bin and 1 indicates 
       a spike in that time bin.
       First dimention should be time and second should be the trial number.
       data: time*trial
       l: time window
    """

    psth = np.mean(data,axis=1)
    length = np.shape(data)[0]

    if np.mod(np.shape(data)[0],l):
        data[length:(length+np.mod(-np.shape(data)[0],l)),:,:] = 0
        psth[length:(length+np.mod(-np.shape(data)[0],l)),:]   = 0

    if len(np.shape(psth))>1:
        dataj = np.squeeze(np.sum(np.reshape(data,[l,np.shape(data)[0]//l,np.shape(data)[1],np.shape(data)[2]], order='F'), axis=0))
        psthj = np.squeeze(np.sum(np.reshape(psth,[l,np.shape(psth)[0]//l,np.shape(psth)[1]], order='F'), axis=0))
        psthj = np.reshape(psthj,[np.shape(psthj)[0],1,np.shape(psthj)[1]], order='F')
        
        psthj[psthj==0] = 10e-10
        corr = dataj/np.tile(psthj,[1, np.shape(dataj)[1], 1]);
        corr = np.reshape(corr,[1,np.shape(corr)[0],np.shape(corr)[1],np.shape(corr)[2]], order='F')
        corr = np.tile(corr,[l, 1, 1, 1])
        corr = np.reshape(corr,[np.shape(corr)[0]*np.shape(corr)[1],np.shape(corr)[2],np.shape(corr)[3]], order='F');
        psth = np.reshape(psth,[np.shape(psth)[0],1,np.shape(psth)[1]], order='F');
        output = np.tile(psth,[1, np.shape(corr)[1], 1])*corr
        output = output[:length,:,:]
        
    elif len(np.shape(psth))==1:
        dataj = np.squeeze(np.sum(np.reshape(data,[l,np.shape(data)[0]//l,np.shape(data)[1]], order='F'), axis=0))
        psthj = np.sum(np.reshape(psth,[l,np.shape(psth)[0]//l], order='F'), axis=0)
        psthj = np.reshape(psthj,[np.shape(psthj)[0],1], order='F')
        
        psthj[psthj==0] = 10e-10
        corr = dataj/np.tile(psthj,[1, np.shape(dataj)[1]])
        corr = np.reshape(corr,[1,np.shape(corr)[0],np.shape(corr)[1]], order='F')
        corr = np.tile(corr,[l, 1, 1])
        corr = np.reshape(corr,[np.shape(corr)[0]*np.shape(corr)[1],np.shape(corr)[2]], order='F');
        psth = np.reshape(psth,[np.shape(psth)[0],1], order='F');
        output = np.tile(psth,[1, np.shape(corr)[1]])*corr
        output = output[:length,:]
        
    if np.shape(data)[0] == l:
        dataj = np.reshape(dataj,[1,np.shape(dataj)[0],np.shape(dataj)[1]], order='F');
        psthj = np.reshape(psthj,[1,np.shape(psthj[0])], order='F');
        
    return output


def get_NFFT_ver1(data):
    """data is version 1 input shape: neuron*ori*rep*time"""
    window=200
    n_t = np.shape(data)[3]
    # triangle function
    t = np.arange(-(n_t-1),(n_t-1))
    theta = n_t-np.abs(t)
    del t
    NFFT = int(nextpow2(2*n_t))
    target = np.array([int(i) for i in NFFT/2+np.arange((-n_t+2),n_t)])
    return window, theta, NFFT, target

def get_ccgshuffle():
    """shuffle corrected ccg with version 1 input shape"""
    ccgshuffle = []
    P=[]
    PP=[]
    pair=0
    for i in np.arange(n_unit-1): # V1 cell
        for m in np.arange(i+1,n_unit):  # V2 cell
            # temp format: [rep, (ori), time], *time has to be the last dim
            if FR[i]>2 and FR[m]>2:
                temp1 = np.squeeze(data[i,:,:,:])
                temp2 = np.squeeze(data[m,:,:,:])
                FR1 = np.squeeze(np.mean(np.sum(temp1,axis=2), axis=1))
                FR2 = np.squeeze(np.mean(np.sum(temp2,axis=2), axis=1))

                tempshift = xcorrfft(temp1[:,:-1,:],temp2[:,1:,:],NFFT)
                tempccg = xcorrfft(temp1,temp2,NFFT)
                tempshift=np.squeeze(np.nanmean(tempshift[:,:,target],axis=1))
                tempccg=np.squeeze(np.nanmean(tempccg[:,:,target],axis=1))
                ccgshuffle.append((tempccg - tempshift).T/np.multiply(np.tile(np.sqrt(FR1*FR2), (len(target), 1)), np.tile(theta.T.reshape(len(theta),1),(1,len(FR1)))))
                P.append([i,m])
                PP.append(pair)
                if pair%10001==0:
                    print(float(pair))
                    print('save')
                    ccgshuffle=np.array(ccgshuffle)
    ccgshuffle=np.array(ccgshuffle)
    return ccgshuffle, P, PP


def get_ccgjitter(spikes, FR, basename, endname, window=500, jitterwindow=25):
    # spikes: neuron*ori*trial*time
    # currently, time need to be devidible by jitterwindow
    assert np.shape(spikes)[0]==len(FR)

    n_unit=np.shape(spikes)[0]
    n_t = np.shape(spikes)[3]
    # triangle function
    t = np.arange(-(n_t-1),(n_t-1))
    theta = n_t-np.abs(t)
    del t
    NFFT = int(nextpow2(2*n_t))
    target = np.array([int(i) for i in NFFT/2+np.arange((-n_t+2),n_t)])

    ccgshuffle = []
    ccgjitter = []
    P=[]
    pair=0
    for i in np.arange(n_unit-1): # V1 cell
        for m in np.arange(i+1,n_unit):  # V2 cell
            # temp format: [rep, (ori), time], *time has to be the last dim
            if FR[i]>2 and FR[m]>2:
                # temp format: [rep, (ori), time], *time has to be the last dim
                # input shape is neuron*ori*rep*time
                temp1 = np.squeeze(spikes[i,:,:,:])
                temp2 = np.squeeze(spikes[m,:,:,:])
                FR1 = np.squeeze(np.mean(np.sum(temp1,axis=2), axis=1))
                FR2 = np.squeeze(np.mean(np.sum(temp2,axis=2), axis=1))

                tempshift = xcorrfft(temp1[:,:-1,:],temp2[:,1:,:],NFFT)
                tempccg = xcorrfft(temp1,temp2,NFFT)
                tempshift=np.squeeze(np.nanmean(tempshift[:,:,target],axis=1))
                tempccg=np.squeeze(np.nanmean(tempccg[:,:,target],axis=1))
                ccgshuffle.append((tempccg - tempshift).T/np.multiply(np.tile(np.sqrt(FR[i]*FR[m]), (len(target), 1)), 
                    np.tile(theta.T.reshape(len(theta),1),(1,len(FR1)))))

                # input shape is time*neuron*ori*rep
                temp1 = np.rollaxis(np.rollaxis(temp1,2,0), 2,1)
                temp2 = np.rollaxis(np.rollaxis(temp2,2,0), 2,1)
                ttemp1 = jitter(temp1,jitterwindow);  
                ttemp2 = jitter(temp2,jitterwindow);
                tempjitter = xcorrfft(np.rollaxis(np.rollaxis(ttemp1,2,0), 2,1),np.rollaxis(np.rollaxis(ttemp2,2,0), 2,1),NFFT);  
                tempjitter = np.squeeze(np.nanmean(tempjitter[:,:,target],axis=1))
                best_ori = np.argmax(np.max((tempccg - tempjitter).T, axis=0))
                ccgjitter.append((tempccg - tempjitter).T/np.multiply(np.tile(np.sqrt(FR[i]*FR[m]), (len(target), 1)), 
                    np.tile(theta.T.reshape(len(theta),1),(1,len(FR1)))))
                P.append([i,m])

                if pair%1001==0:
                    print(float(pair))
                    print('save')
                    ccgshuffle=np.array(ccgshuffle)
                    ccgjitter=np.array(ccgjitter)
                    if i>100:
                        filename=basename+'sc_'+str(i)+endname
                        np.save(filename, ccgshuffle[:,winall/2-window:winall/2+window+1,:])
                        filename=basename+'jc_'+str(i)+endname
                        np.save(filename, ccgjitter[:,winall/2-window:winall/2+window+1,:])
                        filename=basename+'id_'+str(i)+endname
                        np.save(filename, P)
                    else:
                        filename=basename+'sc_0'+str(i)+endname
                        np.save(filename, ccgshuffle[:,winall/2-window:winall/2+window+1,:])
                        filename=basename+'jc_0'+str(i)+endname
                        np.save(filename, ccgjitter[:,winall/2-window:winall/2+window+1,:])
                        filename=basename+'id_0'+str(i)+endname
                        np.save(filename, P)
                    ccgshuffle = []
                    ccgjitter = []
                    P=[]
                

    ccgshuffle=np.array(ccgshuffle)
    ccgjitter = np.array(ccgjitter)
    P=np.array(P)

    filename=basename+'sc_'+'900'+endname
    np.save(filename, ccgshuffle[:,winall/2-window:winall/2+window+1,:])
    filename=basename+'jc_'+'900'+endname
    np.save(filename, ccgjitter[:,winall/2-window:winall/2+window+1,:])
    filename=basename+'id_'+'900'+endname
    np.save(filename, P)
    return 'complete'



#----------selecting and plotting CCG
def filter_ccg(data, plot=False):
    """Separate out broad band CCG from sharp peak CCG"""
    # high pass filter to get the sharp peak of CCG, with cutoff freq=5
    sample_rate = 1000 #Hz
    step_size = 1.0 / sample_rate
    L = len(data)
    time_vec = np.arange(L) * step_size

    ft = scipy.fftpack.fft(data)

    # Lets design a high pass filter to reject high frequency noise above 5Hz
    cutoff_freq = 10
    low = cutoff_freq / (0.5 * sample_rate) 
    order = 7  
    b, a = signal.butter(order, [low], btype='high')

    # With parameters (b, a), we can plot the frequency response of this filter 
    w, h = signal.freqz(b, a)
    filter_freq_axis  = 1 / (2 * step_size * np.pi) * w
    filter_response = abs(h)

    #plt.figure(figsize=(12, 5))
    #plt.plot(pos_freqs, pos_ft / pos_ft.max())
    #plt.plot(filter_freq_axis, filter_response, 'red');
    #plt.xlabel('Frequency (Hz)');

    # Finally, we can filter our signal and see the result:
    sharp_ccg = signal.filtfilt(b, a, data) # *see note below
    broad_ccg = data-sharp_ccg
    if plot==True:
        plt.figure(figsize=(12, 5))
        plt.plot(time_vec, data)
        plt.plot(time_vec, sharp_ccg);
        plt.xlabel('s');
    return time_vec, sharp_ccg, broad_ccg

def select_peak(data, fold=5):
    # for +/500
    tmp = np.max(np.real(data[:,400:600]), axis=1)
    tmpp = np.std(np.real(data[:,np.concatenate([np.arange(100), np.arange(901,1001)], axis=0)]), axis=1)
    thresh = tmpp*fold
    index = np.where(tmp - thresh>0)[0]
    print(len(index))
    return index
    
def filter_ccg_all(data):
    """Filter each ccg before average."""
    ccg=np.zeros_like(data)
    for i in range(np.shape(data)[0]):
        for ori in range(8):
            tmp = data[i,:,ori]
            ccg[i,:,ori] = filter_ccg(tmp)[1]
    return ccg


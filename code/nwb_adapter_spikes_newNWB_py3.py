# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 11:03:40 2016

@author: Xiaoxuan Jia
"""

# utils for spikes analysis
import numpy as np
from scipy import stats
import platform
if int(platform.python_version()[0])>2:
    import _pickle as pk
else:
    import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew


def find_range(x,a,b,option='within'):
    """Find data within range [a,b]"""
    if option=='within':
        return np.where(np.logical_and(x>=a, x<=b))
    elif option=='outside':
        return np.where(np.logical_or(x < a, x > b))
    else:
        print('undefined function')

def get_PSTH(unit_binarized, PSTH_bintime, fs=1000):
    """Calculate PSTH averaged across trials based on binarized spike trains.
    unit_binarized: units*rep*time
    PSTH_bintime: 
    """
    PSTH_binsize = int(PSTH_bintime/1000.*fs)
    sizes = np.shape(unit_binarized)
    print(sizes)

    if len(sizes)==4:
        unit_psth=np.zeros([sizes[0],sizes[1],sizes[2], int(sizes[3]/PSTH_binsize)])
        
        for u in range(sizes[0]):
            for o in range(sizes[1]):
                for r in range(sizes[2]):
                    data = unit_binarized[u,o,r,:]
                    psth = np.sum(np.reshape(data,(int(np.shape(unit_binarized)[3]/PSTH_binsize),PSTH_binsize)),axis=1)
                    unit_psth[u,o,r,:]=psth
        time = np.array(range(len(psth)))*PSTH_bintime/1000.

    if len(sizes)==3:
        unit_psth=np.zeros([sizes[0],sizes[1],int(sizes[2]/PSTH_binsize)])
        
        for u in range(sizes[0]):
            for r in range(sizes[1]):
                data = unit_binarized[u,r,:]
                psth = np.sum(np.reshape(data,(int(np.shape(unit_binarized)[2]/PSTH_binsize),PSTH_binsize)),axis=1)
                unit_psth[u,r,:]=psth
        time = np.array(range(len(psth)))*PSTH_bintime/1000.

    if len(sizes)==2:
        unit_psth=np.zeros([sizes[0],sizes[1]/PSTH_binsize])
        
        for u in range(sizes[0]):
            data = unit_binarized[u,:]
            psth = np.sum(np.reshape(data,(int(np.shape(unit_binarized)[1]/PSTH_binsize),PSTH_binsize)),axis=1)
            unit_psth[u,:]=psth
        time = np.array(range(len(psth)))*PSTH_bintime/1000.

    return unit_psth, time

def plot_PSTH(psth, time, unit_list):
    """Plot calculated psth
    psth is unit by time (averaged across repeats)
    """
    plt.figure(figsize=(20,20))
    for i in range(np.shape(psth)[0]):
        ax = plt.subplot(10,round(np.shape(psth)[0]/10)+1,i+1)
        ax.plot(time,psth[i,:])
        ax.set_title(str(unit_list[i]))
        if i < np.shape(psth)[0]-1:
            ax.set_xticks([])
    return 'end'

def get_FR(unit_binarized, ISI, interval, delay=70):
    """Calculate FR. ISI is stimulus presentation time."""
    # Threshold for FR
    ISI_ms = int(round(ISI*1000))
    if ISI_ms+delay<=np.shape(unit_binarized)[2]:
        FR = np.sum(unit_binarized[:,:,delay:(ISI_ms+delay)], axis=2)/float(ISI_ms)*1000
    else:
        FR = np.sum(unit_binarized[:,:,delay:], axis=2)/float(np.shape(unit_binarized)[2]-delay)*1000
    # consider information delay and remove rebound effect
    if interval!=0:
        FR_ISI = np.sum(unit_binarized[:,:,(ISI_ms+200):], axis=2)/float(np.shape(unit_binarized)[2]--ISI_ms-200)*1000
    else:
        FR_ISI=0
    return FR, FR_ISI

def get_FR_baseline_subtracted(unit_binarized, ISI, interval, delay=70):
    """Calculate FR. ISI is stimulus presentation time."""
    # Threshold for FR
    # subtracted baseline activity in each trial
    ISI_ms = int(round(ISI*1000))
    if ISI_ms+delay<=np.shape(unit_binarized)[2]:
        FR = np.sum(unit_binarized[:,:,delay:(ISI_ms+delay)], axis=2)/float(ISI_ms)*1000-np.sum(unit_binarized[:,:,:delay], axis=2)/float(delay)*1000
    else:
        FR = np.sum(unit_binarized[:,:,delay:], axis=2)/float(np.shape(unit_binarized)[2]-delay)*1000-np.sum(unit_binarized[:,:,:delay], axis=2)/float(delay)*1000
    # consider information delay and remove rebound effect
    if interval!=0:
        FR_ISI = np.sum(unit_binarized[:,:,(ISI_ms+200):], axis=2)/float(np.shape(unit_binarized)[2]-ISI_ms-200)*1000
    else:
        FR_ISI=0
    return FR, FR_ISI

class NWB_adapter(object):

    def __init__(self, nwb_data):
        self.nwb_data = nwb_data
        self.probe_list = list(nwb_data['processing'].keys())
        self.stim_list = list(nwb_data['stimulus']['presentation'].keys())

    def get_unit_meta(self, probename):
        """get relative depth of all sorted units on the probe."""
        unit_list=np.array(list(self.nwb_data['processing'][probename]['UnitTimes'].keys()))[:-1] # because unit_list is in list
        if '.' not in unit_list[0]:
            unit_list=self.nwb_data['processing'][probename]['unit_list'].value.astype(int)

        channel_list=[]
        ypos_list=[]
        snr_list=[]
        ccf_list=[]
        probe_ids=[]
        #waveform_type=[]
        for u in unit_list:
            channel_list.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['channel'].value)
            ypos_list.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['ypos_probe'].value)
            snr_list.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['snr'].value)
            ccf_list.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['ccf_structure'].value)
            probe_ids.append(probename)
            #waveform_type.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['snr'].value)
        channel_list=np.array(channel_list)
        ypos_list=np.array(ypos_list)
        snr_list=np.array(snr_list)
        ccf_list=np.array(ccf_list)
        probe_ids=np.array(probe_ids)

        df=pd.DataFrame()
        df['unit_id']=unit_list
        df['channel_id']=channel_list
        df['ypos']=ypos_list
        df['snr']=snr_list
        df['ccf']=ccf_list
        df['probe_id']=probe_ids
        return df
        
    def get_unit_meta_qc(self, probename):
        """
        added columns for qc matrix
        """
        unit_list=np.array(list(self.nwb_data['processing'][probename]['UnitTimes'].keys()))[:-1] # because unit_list is in list
        if '.' not in unit_list[0]:
            unit_list=self.nwb_data['processing'][probename]['unit_list'].value.astype(int)

        channel_list=[]
        ypos_list=[]
        snr_list=[]
        ccf_list=[]
        probe_ids=[]

        qc_amp=[]
        qc_dprime=[]
        qc_fr=[]
        qc_isi_viol=[]
        qc_iso_dist=[]
        qc_max_drift=[]
        qc_nn_hr=[]
        qc_presence_ratio=[]
        qc_sil_score=[]
        #waveform_type=[]
        for u in unit_list:
            channel_list.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['channel'].value)
            ypos_list.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['ypos_probe'].value)
            ccf_list.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['ccf_structure'].value)
            
            qc_amp.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['amplitude_cutoff'].value)
            qc_dprime.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['d_prime'].value)
            qc_fr.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['firing_rate'].value)
            qc_isi_viol.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['isi_viol'].value)
            qc_iso_dist.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['isolation_distance'].value)
            qc_max_drift.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['max_drift'].value)
            qc_nn_hr.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['nn_hit_rate'].value)
            qc_presence_ratio.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['presence_ratio'].value)
            qc_sil_score.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['silhouette_score'].value)

            if 'snr' in list(self.nwb_data['processing'][probename]['UnitTimes'][str(u)].keys()):
                snr_list.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['snr'].value)
            else:
                snr_list.append(np.NaN)
            probe_ids.append(probename)
            #waveform_type.append(self.nwb_data['processing'][probename]['UnitTimes'][str(u)]['snr'].value)
        channel_list=np.array(channel_list)
        ypos_list=np.array(ypos_list)
        snr_list=np.array(snr_list)
        ccf_list=np.array(ccf_list)
        probe_ids=np.array(probe_ids)

        qc_amp=np.array(qc_amp)
        qc_dprime=np.array(qc_dprime)
        qc_fr=np.array(qc_fr)
        qc_isi_viol=np.array(qc_isi_viol)
        qc_iso_dist=np.array(qc_iso_dist)
        qc_max_drift=np.array(qc_max_drift)
        qc_nn_hr=np.array(qc_nn_hr)
        qc_presence_ratio=np.array(qc_presence_ratio)
        qc_sil_score=np.array(qc_sil_score)

        df=pd.DataFrame()
        df['unit_id']=unit_list
        df['channel_id']=channel_list
        df['ypos']=ypos_list
        df['snr']=snr_list
        df['ccf']=ccf_list
        df['probe_id']=probe_ids
        df['qc_amp']=qc_amp
        df['qc_dprime']=qc_dprime
        df['qc_fr']=qc_fr
        df['qc_isi_viol']=qc_isi_viol
        df['qc_iso_dist']=qc_iso_dist
        df['qc_max_drift']=qc_max_drift
        df['qc_nn_hr']=qc_nn_hr
        df['qc_presence_ratio']=qc_presence_ratio
        df['qc_sil_score']=qc_sil_score
        return df

    def get_stim_table(self, key):
        """Get stimulus sync pulse for specified (key) stimuli.
        Opto tagging exp only has one column: 'Start'. 
        """
        if key!='spontaneous':
            temp1 = pd.DataFrame(self.nwb_data['stimulus']['presentation'][key]['data'].value, columns = self.nwb_data['stimulus']['presentation'][key]['features'].value)
            if min(np.shape(self.nwb_data['stimulus']['presentation'][key]['timestamps'].value))==2:
                temp2 = pd.DataFrame(self.nwb_data['stimulus']['presentation'][key]['timestamps'].value, columns = ['Start','End'])
            else:
                temp2 = pd.DataFrame(self.nwb_data['stimulus']['presentation'][key]['timestamps'].value, columns = ['Start'])
            stim_table = temp2.join(temp1)
            del temp1, temp2
        else:
            stim_table = pd.DataFrame(self.nwb_data['stimulus']['presentation'][key]['timestamps'].value, columns = ['Start','End'])     
        return stim_table

    def get_spike_times(self, probename, unit):
        #print(unit)
        if type(unit)==str:
            return np.array(self.nwb_data['processing'][probename]['UnitTimes'][unit]['times'].value)  
        if type(unit)==bytes:
            return np.array(self.nwb_data['processing'][probename]['UnitTimes'][unit]['times'].value)  
        if type(unit)!=str and type(unit)!=bytes:
            return np.array(self.nwb_data['processing'][probename]['UnitTimes'][str(int(unit))]['times'].value)  
        
    def get_ISI(self, stim_table):
        """Inter-stimulus interval for consistent size of binary array. Because there is 
        slight variability in ISI. """
        if 'End' in list(stim_table.keys()):
            ISI = round(np.median(stim_table['End'].values - stim_table['Start'].values), 3)
        else:
            ISI = round(np.median(np.diff(stim_table['Start'].values)), 3)     
        return ISI   

    def get_interval(self, stim_table):
        if 'End' in list(stim_table.keys()):
            self.interval = np.mean(stim_table.Start[1:].values-stim_table.End[:-1].values)
        else:
            self.interval = 0
        return self.interval

    def get_binarized(self, probename, unit, stim_table, pre_time=0., post_time=0., fs = 1000.):
        """Binarize spike trains with 1/fs ms window.
        """
        ISI = self.get_ISI(stim_table)
        sync_start = stim_table['Start']-pre_time
        sync_end = stim_table['Start']+ISI+post_time

        for i in range(20):
            time_range = int(round(max(sync_end.values-sync_start.values)*fs)+i)
            if time_range%10 ==0:
                #print(time_range)
                #print(i,temp)
                break

        spike_times = self.get_spike_times(probename, unit)
        #print(unit, type(unit))
        time_repeats = []
        spike_times = spike_times*fs # convert to ms
        for i,t in enumerate(sync_start*fs):
            temp = spike_times[find_range(spike_times,t,t+time_range)]-t  
            time_repeats.append(temp/fs)

        time_repeats=np.array(time_repeats)

        binarized = []
        for trial in time_repeats:
            binarized_ = np.zeros(time_range) # variability in lenth
            for spike in trial:
                spike_t = int(np.floor(spike*fs))
                if spike_t<time_range:
                    binarized_[spike_t] = 1
            binarized.append(binarized_)   
        bin_times = 1./fs*np.arange(time_range)
        return time_repeats, binarized, bin_times

    def get_binarized_spon(self, probename, unit, sync_start, sync_end,fs = 1000.):
        """For spontaneous activity, for given single start and end time. 
        Convert timebase to ms.
        """
        for i in range(20):
            time_range = int(round(sync_end-sync_start)*fs)+i
            if time_range%10 ==0:
                #print(time_range)
                #print(i,temp)
                break

        spike_times = self.get_spike_times(probename, unit)

        t = sync_start*fs
        spike_times = spike_times*fs
        temp = spike_times[find_range(spike_times,t,t+time_range)]-t

        binarized_ = np.zeros(time_range) # variability in lenth
        for spike in temp:
            spike_t = int(np.floor(spike))
            if spike_t<time_range:
                binarized_[spike_t] = 1
        return binarized_

    def get_probe_V1(self, probename, key, wm, stim_table=[], pre_time=0., post_time=0., fs = 1000.):
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
        ISI = self.get_ISI(stim_table)

        print(probename)
        df=self.get_unit_meta(probename)
        # sort according to channel/depth
        df_sorted = df.sort_values(['channel_id'])

        unit_list = df_sorted.unit_id.values.flatten()
        channel_ypos = df_sorted.ypos.values.flatten()
        channel_list = df_sorted.channel_id.values.flatten()
        snr_list = df_sorted.snr.values.flatten()

        unit_binarized = []
        unit_list_V1=[]
        channel_list_V1=[]
        channel_ypos_V1=[]
        snr_v1=[]

        for idx, unit in enumerate(unit_list):
            ypos = channel_ypos[idx]
            if ypos>wm:
                time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                unit_binarized.append(binarized)
                unit_list_V1.append(unit_list[idx])
                channel_list_V1.append(channel_list[idx])
                channel_ypos_V1.append(ypos)
                snr_v1.append(snr_list[idx])          

        # todo: can be replaced with xarray to label matrix
        unit_binarized = np.array(unit_binarized)
        unit_list_V1=np.array(unit_list_V1)
        channel_list_V1=np.array(channel_list_V1)
        channel_ypos_V1=np.array(channel_ypos_V1)
        snr_v1=np.array(snr_v1)
        return unit_binarized, unit_list_V1, channel_list_V1, channel_ypos_V1, snr_v1

    def get_probe_ccf(self, probename, key, target_structure='VIS', stim_table=[], pre_time=0., post_time=0., fs = 1000.):
        """
        target_structure: string of ccf name
        """
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
        ISI = self.get_ISI(stim_table)

        print(probename)
        df=self.get_unit_meta(probename)
        # sort according to channel/depth
        df_sorted = df.sort_values(['channel_id'])

        unit_list = df_sorted.unit_id.values.flatten()
        channel_ypos = df_sorted.ypos.values.flatten()
        channel_list = df_sorted.channel_id.values.flatten()
        snr_list = df_sorted.snr.values.flatten()
        ccf_list = df_sorted.ccf.values.flatten()

        unit_binarized = []
        unit_list_V1=[]
        channel_list_V1=[]
        channel_ypos_V1=[]
        snr_v1=[]

        for idx, unit in enumerate(unit_list):
            ccf = str(ccf_list[idx])
            if target_structure in ccf:
                time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                unit_binarized.append(binarized)
                unit_list_V1.append(unit_list[idx])
                channel_list_V1.append(channel_list[idx])
                channel_ypos_V1.append(channel_ypos[idx])
                snr_v1.append(snr_list[idx])          

        # todo: can be replaced with xarray to label matrix
        unit_binarized = np.array(unit_binarized)
        unit_list_V1=np.array(unit_list_V1)
        channel_list_V1=np.array(channel_list_V1)
        channel_ypos_V1=np.array(channel_ypos_V1)
        snr_v1=np.array(snr_v1)
        return unit_binarized, unit_list_V1, channel_list_V1, channel_ypos_V1, snr_v1

    def get_matrix_V1_sortch(self, key, pre_time=0., post_time=0., wm_given={}, stim_table=[], probenames=[], fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time."""
        # cancatenate all data from different probes
        if len(probenames)==0:
            probenames = list(self.nwb_data['processing'].keys())
            
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')

        unit_binarized = []
        unit_list_V1 = []
        channel_V1 = []
        #waveforms=[]
        snr_v1=[]
        probe_ids=[]
        for probename in probenames:
            print(probename)

            df=self.get_unit_meta(probename)
            # sort according to channel/depth
            df_sorted = df.sort_values(['channel_id'])

            unit_list = df_sorted.unit_id.values.flatten()
            channel_ypos = df_sorted.ypos.values.flatten()
            channel_list = df_sorted.channel_id.values.flatten()
            snr_list = df_sorted.snr.values.flatten()

            wm = wm_given[probename]

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(unit_list):
                ypos = channel_ypos[idx]
                if ypos>wm:
                    time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                    unit_binarized.append(binarized)
                    unit_list_V1.append(unit)
                    channel_V1.append(channel_list[idx])
                    #waveforms.append(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['template'].value) 
                    snr_v1.append(snr_list[idx])
                    probe_ids.append(probename)
        
        unit_binarized = np.array(unit_binarized, dtype=np.uint8)
        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        #waveforms = np.array(waveforms)
        snr_v1=np.array(snr_v1)
        probe_ids=np.array(probe_ids)
        return unit_binarized, unit_list_V1, channel_V1, snr_v1, probe_ids

    def get_matrix_ccf_sortch(self, key, target_structure='VIS', pre_time=0., post_time=0., stim_table=[], probenames=[], fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time."""
        # cancatenate all data from different probes
        if len(probenames)==0:
            probenames = list(self.nwb_data['processing'].keys())
            
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')

        unit_binarized = []
        unit_list_V1 = []
        channel_V1 = []
        ypos=[]
        #waveforms=[]
        ccf_v1=[]
        snr_v1=[]
        probe_ids=[]

        for probename in probenames:
            print(probename)

            df=self.get_unit_meta(probename)
            # sort according to channel/depth
            df_sorted = df.sort_values(['channel_id'])

            unit_list = df_sorted.unit_id.values.flatten()
            channel_ypos = df_sorted.ypos.values.flatten()
            channel_list = df_sorted.channel_id.values.flatten()
            snr_list = df_sorted.snr.values.flatten()
            ccf_list = df_sorted.ccf.values.flatten()

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(unit_list):
                ccf = str(ccf_list[idx])
                if target_structure in ccf:
                    time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                    unit_binarized.append(binarized)
                    unit_list_V1.append(unit)
                    channel_V1.append(channel_list[idx])
                    #waveforms.append(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['template'].value) 
                    ccf_v1.append(ccf_list[idx])
                    snr_v1.append(snr_list[idx])
                    probe_ids.append(probename)
                    ypos.append(channel_ypos[idx])
        
        unit_binarized = np.array(unit_binarized, dtype=np.uint8)
        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        ypos=np.array(ypos)
        #waveforms = np.array(waveforms)
        snr_v1=np.array(snr_v1)
        probe_ids=np.array(probe_ids)
        ccf_v1=np.array(ccf_v1)

        return unit_binarized, unit_list_V1, channel_V1, snr_v1, probe_ids, ccf_v1, ypos


    def get_matrix_V1_sortch_snr(self, key, pre_time=0., post_time=0., wm_given={}, stim_table=[], probenames=[], snr_threshold=1.5, fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time.
        cancatenate all data in cortex from different probes
        sort by depth
        threshold with snr
        """
        # 
        if len(probenames)==0:
            probenames = list(self.nwb_data['processing'].keys())
            
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')

        unit_binarized = []
        unit_list_V1 = []
        channel_V1 = []
        waveforms=[]
        snr_v1=[]
        probe_ids=[]
        for probename in probenames:
            print(probename)

            df=self.get_unit_meta(probename)
            # sort according to channel/depth
            df_sorted = df.sort_values(['channel_id'])

            unit_list = df_sorted.unit_id.values.flatten()
            channel_ypos = df_sorted.ypos.values.flatten()
            channel_list = df_sorted.channel_id.values.flatten()
            snr_list = df_sorted.snr.values.flatten()

            wm = wm_given[probename]

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(unit_list):
                ypos = channel_ypos[idx]
                snr = snr_list[idx]
                if ypos>wm:
                    if snr>=snr_threshold:
                        time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                        unit_binarized.append(binarized)
                        unit_list_V1.append(unit)
                        channel_V1.append(channel_list[idx])
                        snr_v1.append(snr_list[idx])
                        probe_ids.append(probename)
        # dtype=np.uint8 save lots of space           
        unit_binarized = np.array(unit_binarized, dtype=np.uint8)
        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        probe_ids=np.array(probe_ids)
        snr_v1=np.array(snr_v1)
        return unit_binarized, unit_list_V1, channel_V1, snr_v1, probe_ids

    def get_matrix_ccf_sortch_snr(self, key, target_structure='VIS', pre_time=0., post_time=0., stim_table=[], probenames=[], snr_threshold=1.5, fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time.
        cancatenate all data in cortex from different probes
        sort by depth
        threshold with snr
        """
        # 
        if len(probenames)==0:
            probenames = list(self.nwb_data['processing'].keys())
            
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')

        unit_binarized = []
        unit_list_V1 = []
        channel_V1 = []
        waveforms=[]
        snr_v1=[]
        probe_ids=[]
        for probename in probenames:
            print(probename)

            df=self.get_unit_meta(probename)
            # sort according to channel/depth
            df_sorted = df.sort_values(['channel_id'])

            unit_list = df_sorted.unit_id.values.flatten()
            channel_ypos = df_sorted.ypos.values.flatten()
            channel_list = df_sorted.channel_id.values.flatten()
            snr_list = df_sorted.snr.values.flatten()
            ccf_list = df_sorted.ccf.values.flatten()

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(unit_list):
                snr = snr_list[idx]
                ccf = str(ccf_list[idx])
                if target_structure in ccf:
                    if snr>=snr_threshold:
                        time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                        unit_binarized.append(binarized)
                        unit_list_V1.append(unit)
                        channel_V1.append(channel_list[idx])
                        probe_ids.append(probename)
        # dtype=np.uint8 save lots of space           
        unit_binarized = np.array(unit_binarized, dtype=np.uint8)
        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        probe_ids=np.array(probe_ids)
        return unit_binarized, unit_list_V1, channel_V1, probe_ids


    def get_ISI_V1_sortch(self, select=False, wm_given={}, key='',  fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time."""
        # cancatenate all data from different probes
        if select==True:
            stim_table = self.get_stim_table(key=key)
            stimulus_gap=stim_table.Start.values[1:]-stim_table.End.values[:-1]
            if len(stim_table)==0:
                stim_table = self.get_stim_table(key=key)
                print('recompute stim table for key')

        probenames = list(self.nwb_data['processing'].keys())
        print(probenames)
        keys = list(self.nwb_data['stimulus']['presentation'].keys())

        #auto = []
        ISI = []
        unit_list_V1 = []
        channel_V1 = []
        snr_v1=[]
        for probename in probenames:
            print(probename)

            df=self.get_unit_meta(probename)
            # sort according to channel/depth
            df_sorted = df.sort_values(['channel_id'])

            unit_list = df_sorted.unit_id.values.flatten()
            channel_ypos = df_sorted.ypos.values.flatten()
            channel_list = df_sorted.channel_id.values.flatten()
            snr_list = df_sorted.snr.values.flatten()

            wm = wm_given[probename]

            # for only units in V1, calculate binned response for each conditions (1 ms resolution)
            for idx, unit in enumerate(unit_list):
                ypos = channel_ypos[idx]
                snr = snr_list[idx]
                if ypos>wm:
                    # spike times for each unit
                    times = self.get_spike_times(probename, unit)
                    if 'Opto' in keys:
                        # remove opto tagging to avoid artifact in spike times
                        stim_table = self.get_stim_table(key='Opto')
                        times = times[np.where(times<stim_table.Start.values[0])[0]]
                    
                    if select==True:
                        # select times according to stim_table
                        times_selected=[]
                        for idx, start in enumerate(stim_table.Start):
                            end = stim_table.End[idx]
                            if end-start>0:
                                times_selected.append(times[np.where((times>=start) & (times<end))[0]])
                        times_selected = np.array([item for sublist in times_selected for item in sublist])
                        #auto.append(autocorr(times_selected))
                        ISI.append(inter_spike_interval(times_selected))
                    else:
                        #auto.append(autocorr(times))
                        ISI.append(inter_spike_interval(times))
                    
                    unit_list_V1.append(unit)
                    channel_V1.append(channel_list[idx])
                    snr_v1.append(snr) 

        unit_list_V1 = np.array(unit_list_V1)
        channel_V1 = np.array(channel_V1)
        snr_v1=np.array(snr_v1)

        S=[]
        for i in range(len(ISI)):
            # larger range for ISI, bias the skewness value to higher
            hist, bins = np.histogram(ISI[i],bins=np.arange(0,0.06,0.001))
            # normalized sum hist to 1
            plt.step(np.arange(0,0.06,0.001), np.concatenate(([hist[0]],hist),axis=0)/float(sum(hist)))
            S.append(skew(hist/float(sum(hist))))
        S=np.array(S)            
        return ISI, S, unit_list_V1, channel_V1, snr_v1
    
    def get_matrix_selected(self, key, df, pre_time=0., post_time=0., stim_table=[], fs=1000):
        """Create response matrix for given stimulus condition (key).
        Matrix is digitized between ['Start']-pre_time and ['Start']+post_time."""
        # cancatenate all data from different probes

        unit_binarized = []
        # for only units in V1, calculate binned response for each conditions (1 ms resolution)
        for index, row in df.iterrows():
            probename = row.probe_id
            unit = row.unit_id
            time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
            unit_binarized.append(binarized)
            #waveforms.append(self.nwb_data['processing'][probename]['UnitTimes'][str(unit)]['template'].value) 
        unit_binarized = np.array(unit_binarized, dtype=np.uint8)
        return unit_binarized

    def get_selected_waveform(self, df):
        #cell_types=[]
        waveforms=[]
        waveforms_2D=[]

        for index, row in df.iterrows():
            #cell_types.append(nwb_data['processing'][row.probe_id]['UnitTimes'][str(row.unit_id)]['type'].value)
            #print(str(row.unit_id))
            #if str(row.unit_id) in list(self.nwb_data['processing'][row.probe_id]['UnitTimes'].keys()):
            waveforms.append(self.nwb_data['processing'][row.probe_id]['UnitTimes'][str(row.unit_id)]['waveform'].value)
            peak_ch = int(self.nwb_data['processing'][row.probe_id]['UnitTimes'][str(row.unit_id)]['channel'].value)

            tmp = self.nwb_data['processing'][row.probe_id]['UnitTimes'][str(int(row.unit_id))]['template'].value
            if tmp.shape[0]<tmp.shape[1]:
                tmp=tmp[:,peak_ch-15:peak_ch+15]
                if tmp.shape[1]==30:
                    waveforms_2D.append(tmp[:60, 0::2].T)
                else:
                    waveforms_2D.append(np.zeros((60,15)))
                waveform2D_noise=np.zeros((60,15))
            else:
                tmp=tmp[peak_ch-15:peak_ch+15, :]
                if tmp.shape[0]==30:
                    waveforms_2D.append(tmp[0::2, :60])
                else:
                    waveforms_2D.append(np.zeros((15,60)))
                waveform2D_noise=np.zeros((15,60))
                
            #else:
                #waveforms.append(waveform_noise)
                #waveforms_2D.append(waveform2D_noise)

        #df['type']=cell_types
        #df['waveform']=waveforms
        #df['waveform2D']=waveforms_2D
        return waveforms, waveforms_2D


    def get_matrix_all(self, key, pre_time=0., post_time=0., fs = 1000., stim_table=[]):
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')
        ISI = self.get_ISI(stim_table)

        probenames = list(self.nwb_data['processing'].keys())
        matrix=np.array([], dtype=np.uint8)
        matrix_unit=np.array([])
        matrix_channel=np.array([])
        df_all=[]
        probe_ids_all=[]

        for probename in probenames:
            print(probename)

            df=self.get_unit_meta(probename)
            # sort according to channel/depth
            df_sorted = df.sort_values(['channel_id'])
            unit_list = df_sorted.unit_id.values.flatten()
            channel_list = df_sorted.channel_id.values.flatten()
            df_all.append(df_sorted)

            unit_binarized = []
            for idx, unit in enumerate(unit_list):
                time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                unit_binarized.append(binarized)
            unit_binarized = np.array(unit_binarized, dtype=np.uint8)
            
            if len(matrix)==0:
                matrix=unit_binarized
                matrix_unit=unit_list
                matrix_channel=channel_list
            else:
                matrix = np.concatenate((matrix, unit_binarized),axis=0)
                matrix_unit = np.concatenate((matrix_unit, unit_list),axis=0)
                matrix_channel = np.concatenate((matrix_channel, channel_list),axis=0)
        df_all=pd.concat(df_all, axis=0)
        return matrix, matrix_unit, matrix_channel, df_all

    def get_matrix_all_qc(self, key, pre_time=0., post_time=0., fs = 1000., stim_table=[]):
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')
        ISI = self.get_ISI(stim_table)

        probenames = list(self.nwb_data['processing'].keys())
        matrix=np.array([], dtype=np.uint8)
        matrix_unit=np.array([])
        matrix_channel=np.array([])
        df_all=[]
        probe_ids_all=[]

        for probename in probenames:
            print(probename)

            df=self.get_unit_meta_qc(probename)
            # sort according to channel/depth
            df_sorted = df.sort_values(['channel_id'])
            unit_list = df_sorted.unit_id.values.flatten()
            channel_list = df_sorted.channel_id.values.flatten()
            df_all.append(df_sorted)

            unit_binarized = []
            for idx, unit in enumerate(unit_list):
                time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                unit_binarized.append(binarized)
            unit_binarized = np.array(unit_binarized, dtype=np.uint8)
            
            if len(matrix)==0:
                matrix=unit_binarized
                matrix_unit=unit_list
                matrix_channel=channel_list
            else:
                matrix = np.concatenate((matrix, unit_binarized),axis=0)
                matrix_unit = np.concatenate((matrix_unit, unit_list),axis=0)
                matrix_channel = np.concatenate((matrix_channel, channel_list),axis=0)
        df_all=pd.concat(df_all, axis=0)
        return matrix, matrix_unit, matrix_channel, df_all

    def get_meta_all_qc(self):

        probenames = list(self.nwb_data['processing'].keys())
        df_all=[]

        for probename in probenames:
            print(probename)
            df=self.get_unit_meta_qc(probename)
            # sort according to channel/depth
            df_sorted = df.sort_values(['channel_id'])
            df_all.append(df_sorted)

        df_all=pd.concat(df_all, axis=0)
        return df_all

    def get_matrix_all_spon(self, key, pre_time=0., post_time=0., fs = 1000., stim_table=[]):
        """To do: add field to select time window for spontaneous activity."""
        if len(stim_table)==0:
            stim_table = self.get_stim_table(key=key)
            print('recompute stim table for key')
        ISI = self.get_ISI(stim_table)

        probenames = list(self.nwb_data['processing'].keys())
        matrix=np.array([], dtype=np.uint8)
        matrix_unit=np.array([])
        matrix_channel=np.array([])
        df_all=[]
        for probename in probenames:
            print(probename)
            df=self.get_unit_meta(probename)
            # sort according to channel/depth
            df_sorted = df.sort_values(['channel_id'])
            unit_list = df_sorted.unit_id.values.flatten()
            df.append(df_sorted)

            unit_binarized = []

            for idx, unit in enumerate(unit_list):
                time_repeats, binarized, bin_time = self.get_binarized(probename, unit, stim_table, pre_time=pre_time, post_time=post_time, fs=fs)
                unit_binarized.append(binarized)
            unit_binarized = np.array(unit_binarized, dtype=np.uint8)
            
            if len(matrix)==0:
                matrix=unit_binarized
                matrix_unit=unit_list
                matrix_channel=channel_list
            else:
                matrix = np.concatenate((matrix, unit_binarized),axis=0)
                matrix_unit = np.concatenate((matrix_unit, unit_list),axis=0)
                matrix_channel = np.concatenate((matrix_channel, channel_list),axis=0)
        df_all=pd.concat(df_all, axis=0)

        return matrix, matrix_unit, matrix_channel, df_all



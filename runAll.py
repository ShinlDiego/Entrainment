import matplotlib.pyplot as plt
import seaborn as sns
from mne.decoding import ReceptiveField
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from NRC import NRC,recordModule,RegularizedRF
from utils import returnFFT, returnPSD,returnSpec
from scipy import stats

srate = 240
tmin,tmax = -0.2,1
expName = 'exp-2'
# chnNames = ['CPZ','PZ','POZ','OZ','P1','P2','P3','P4','P7','PO3','PO4','PO7','PO8','O1','O2']

chnNames = ['CB1', 'CB2', 'O1', 'OZ', 'O2', 'PO7', 'PO5',
            'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'P7', 'P5',
            'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8','TP7',
            'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8']

dir = './datasets/%s.pickle'%expName

with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

for sub in tqdm(wholeset):

    chnINX = [sub['channel'].index(i) for i in chnNames]

    X = sub['X'][:,chnINX]
    Resting = sub['restX'][:, chnINX]
    S = sub['stimulus']
    
    # RF using merely xorr
    decoder = NRC(srate=srate, tmin=tmin, tmax=tmax, alpha=0.95)

    # decoder = RegularizedRF(srate=srate, tmin=tmin, tmax=tmax, mTRF=False, alpha=1e10)
    decoder.fit(R=X, S=sub['stimulus'])

    # record TRF
    recoder = recordModule(srate=srate,sub=sub['name'],chn=chnNames,exp=expName)
    csr = decoder.Csr[:,:,np.newaxis,:]
    recoder.recordKernel(csr, sub['y'], 'uni', tmin, tmax)

    # record TRF spectral
    # kernel = decoder.kernel.squeeze()

    # kernel_1 = np.mean(kernel[:100],axis=0,keepdims=True)
    # kernel_2 = np.mean(kernel[100:200],axis=0,keepdims=True)
    # kernel_3 = np.mean(kernel[-100:],axis=0,keepdims=True)
    # kernel = np.concatenate([kernel_1,kernel_2,kernel_3])

    # freqz, F_k = returnPSD(kernel)
    # F_k = np.mean(F_k, axis=1, keepdims=True)
    # F_k = stats.zscore(F_k, axis=-1)
    # recoder.recordSpectral(freqz, F_k,sub['y'], 'TRF')
    
    # # recording resting EEG
    # freqz, F_rest = returnPSD(Resting)
    # F_rest = np.mean(F_rest, axis=1, keepdims=True)
    # F_rest = stats.zscore(F_rest,axis=-1)
    # recoder.recordSpectral(freqz, F_rest, [0, 0], 'resting')

    # # recording EEG 
    # freqz, F_X = returnPSD(X)
    # F_X = np.mean(F_X, axis=1, keepdims=True)
    # F_X = stats.zscore(F_X, axis=-1)
    # recoder.recordSpectral(freqz, F_X, sub['y'], 'EEG')

    # freqz, F_S = returnPSD(S)
    # F_S = stats.zscore(F_S, axis=-1)
    # recoder.recordSpectral(freqz, F_S[:,np.newaxis,:], sub['y'], 'STI')

    # record temporal EEG
    # recoder.recordEEG(X,sub['y']) 

    # record stimimulus
    # recoder.recordStimulus(sub['stimulus'],sub['y'])

    # recoder.recordKernel(decoder.Csr,sub['y'],'unwhitened',tmin,tmax)

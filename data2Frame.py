import matplotlib.pyplot as plt
from scipy.signal import hilbert
import seaborn as sns
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from utils import *
from scipy import stats


srate=500
expName = 'exp-1'
subExp = 'ssvep'
picks = ['O1', 'OZ', 'O2','POZ','PZ','PO3','PO4','PO5','PO6']

dir = './datasets/%s.pickle'%expName

with open(dir, "rb") as fp:
    wholeset = pickle.load(fp)

for sub in tqdm(wholeset):
            
    subName = sub['name']

    chnNames = sub['channel']
    chnINX = [chnNames.index(i) for i in picks]

    X = sub[subExp][0][:, chnINX]
    y = sub[subExp][1]
    
    recorder = recordModule(sub=subName, srate=srate,exp=expName+os.sep+subExp,chn=picks)

    # record raw EEG
    # recorder.recordEEG(X,y)

    # freqz, F_S = returnPSD(X, srate=srate)
    # recorder.recordSpectral(freqz, F_S, y)

    snr,freqz = narrow_snr(X,srate=srate)
    recorder.returnSNR(freqz, snr, y)






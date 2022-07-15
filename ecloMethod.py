from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math

def getChannels(data, chn_picks):
    
    picks = []
    for pickINX, pick in enumerate(data['channel']):
        if pick in chn_picks:
            picks.append(pickINX)
    
    eeg = data['X']
    eeg = sum(eeg[:,pickINX,:] for pickINX in picks)/len(picks)
    
    return eeg

def getAlpha(eeg, fs=1000, nperseg=2000, nfft=5000):
    
    alpha_f = []
    for i in range(eeg.shape[0]):
        f, Pxx = signal.welch(eeg[i,:], fs=fs, nperseg=nperseg, nfft=nfft)
        f = f[25:]
        Pxx = Pxx[25:]
        maxindex = np.argmax(Pxx)
        max_f = round(f[maxindex],1)
        if max_f <=12 and max_f >=8:
            alpha_f.append(max_f)
            
    alpha_bar = {}
    for i in alpha_f:
        alpha_bar[i] = alpha_f.count(i)
        
    fig,ax = plt.subplots()     
    plt.bar(alpha_bar.keys(), alpha_bar.values(), width=0.1)
    ax.set_xticks(list(alpha_bar.keys()))
    
    alpha_sum = 0
    alpha_count = 0
    for i in range(len(list(alpha_bar.keys()))):
        alpha_sum = alpha_sum + list(alpha_bar.keys())[i]*list(alpha_bar.values())[i]
        alpha_count = alpha_count + list(alpha_bar.values())[i]
    alpha = alpha_sum/alpha_count
    
    return alpha

################################################################################################# 
# Evoked power methods
################################################################################################# 
def Blockmean(eeg, Trigger):
    
    trigger = np.unique(Trigger)
    eeg_mean = np.zeros((len(trigger),eeg.shape[1]))
    
    for tg in trigger:
        count = 0
        for trialINX, tritg in enumerate(Trigger):
            if tritg == tg:
                count = count + 1
                eeg_mean[tg-1,:] = eeg_mean[tg-1,:] + eeg[trialINX,:]
        eeg_mean[tg-1,:] = eeg_mean[tg-1,:]/count
    
    return eeg_mean

def EPlotspectrum(eeg, frequency, intensity, alpha, freq, intens, nperseg=5000, nfft=5000, all=False, fs=1000):
    
    if all == False:
        
        fig,ax = plt.subplots()
        
        intensity = np.array([int(ins*10) for ins in intensity])
        Findex = np.where(frequency==freq)[0][0]
        Iindex = np.where(intensity==int(intens*10))[0][0]
        tgindex = Findex*len(intensity)+Iindex
        f, Pxx = signal.welch(eeg[tgindex,:], fs, nperseg=nperseg, nfft=nfft)
        plt.plot(f, Pxx)
        plt.xlim(0,40)  
        plt.show()  
    else:
        trigger = np.arange(len(frequency)*len(intensity))
        stim_FI = np.zeros((len(trigger),2))
        
        for tg in trigger:
            stim_FI[tg, 0] = frequency[math.ceil((tg+1)/len(intensity))-1];
            stim_FI[tg, 1] = intensity[tg-math.ceil((tg+1)/len(intensity))*len(intensity)+len(intensity)];
            
        if len(trigger) == 25:
            fig,axes = plt.subplots(figsize=(10,10), nrows=5, ncols=5)
        else:
            fig,axes = plt.subplots(figsize=(2*len(frequency),2*len(intensity)), nrows=len(intensity), ncols=len(frequency))
            
        for (ax,tg) in zip(axes.flatten(),trigger):
            f, Pxx = signal.welch(eeg[tg,:], fs, nperseg=nperseg, nfft=nfft)
            ax.plot(f, Pxx, label=str(format(stim_FI[tg,0],'.1f'))+','+str(format(stim_FI[tg,1],'.1f')))
            ax.legend()
            ax.set_xlim(0,40)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.vlines(x=float(format(alpha,'.2f')), ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='dashed', colors='k')
            ax.vlines(x=stim_FI[tg,0], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='dashed', colors='r')
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, hspace=0.2, wspace=0.2)
        plt.show()
    
def EPlotHotmap(eeg, frequency, intensity, intens=1, nperseg=5000, nfft=5000, fs=1000, vmax=None, uplim=20):
    
    fig,ax = plt.subplots(figsize=(7,5))
    intensity = np.array([int(ins*10) for ins in intensity])
    dataplot = []
    
    for freq in frequency:
        Findex = np.where(frequency==freq)[0][0]
        Iindex = np.where(intensity==int(intens*10))[0][0]
        tgindex = Findex*len(intensity)+Iindex
        f, Pxx = signal.welch(eeg[tgindex,:], fs, nperseg=nperseg, nfft=nfft)
        dataplot.append(Pxx)
    dataplot = np.stack(dataplot)
    dataplot = dataplot.T
    
    if vmax == None:
        plt.pcolormesh(frequency, f, dataplot, shading='gouraud')
    else:
        plt.pcolormesh(frequency, f, dataplot, shading='gouraud', vmax=vmax)
    plt.ylim(0,uplim)
    x = range(int(frequency[0]),int(frequency[len(frequency)-1]),1)
    y = range(0,uplim,1)
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('Stimulation Frequency\Hz')
    plt.ylabel('Response Frequency\Hz')
    plt.colorbar()
    plt.show()
    
def EPlotTFR(eeg, frequency, intensity, alpha, freq, intens=1, nperseg=1000, nfft=5000, fs=1000, vmax=None, uplim=20):
    
    fig,ax = plt.subplots(figsize=(7,5))
    
    intensity = np.array([int(ins*10) for ins in intensity])
    Findex = np.where(frequency==freq)[0][0]
    Iindex = np.where(intensity==int(intens*10))[0][0]
    tgindex = Findex*len(intensity)+Iindex
    
    f, t, Sxx = signal.spectrogram(eeg[tgindex,:], fs, nperseg=nperseg, nfft=nfft)
    if vmax == None:
        plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='Reds')
    else:
        plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='Reds', vmax=vmax)
    plt.hlines(y=alpha, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles='dashed', colors='k')
    plt.hlines(y=freq, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles='dashed', colors='b')
    plt.colorbar()
    plt.ylim(2,uplim)
    plt.show()

def EPlotpeakratio(eeg, frequency, intensity, alpha, intens=1, nperseg=5000, nfft=5000, fs=1000):
    
    data_alpha = []
    intensity = np.array([int(ins*10) for ins in intensity])
    fig,ax = plt.subplots()

    for freq in frequency:
        Findex = np.where(frequency==freq)[0][0]
        Iindex = np.where(intensity==int(intens*10))[0][0]
        tgindex = Findex*len(intensity)+Iindex
        f, Pxx = signal.welch(eeg[tgindex,:], fs, nperseg=nperseg, nfft=nfft)
        count = 0
        sum_alpha = 0
        
        for findex, f_alpha in enumerate(f):
            if f_alpha<= alpha+0.4 and f_alpha>= alpha-0.4:
                sum_alpha = sum_alpha + Pxx[findex]/Pxx.mean()
                count = count + 1
        data_alpha.append(sum_alpha/count)
        
    data_alpha = np.stack(data_alpha)
    data_alpha = data_alpha/data_alpha.mean()
    frequency = frequency/alpha
    
    plt.plot(frequency, data_alpha, marker='o', color='k')
    plt.xlabel('Stimulation frequency in multiples of alpha')
    plt.ylabel('Alpha peak ratio')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.hlines(y=1, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dashed', color='k')

#################################################################################################   
# Single Trial methods
################################################################################################# 
def SPlotPLV(eeg, frequency, intensity, Trigger, stimulus, latency=0.14, fs=1000):
    
    trilen = len(frequency)*len(intensity)
    
    if trilen == 25:
        PLV = []
        
        for i in range(eeg.shape[0]):
            eeg_phase = np.unwrap(np.angle(signal.hilbert(eeg[i,int(latency*fs):])))
            stim_phase = np.unwrap(np.angle(signal.hilbert(stimulus[i,:stimulus.shape[1]-int(latency*fs)])))
            plv = np.abs(sum(np.exp(1j*(stim_phase-eeg_phase)))/eeg.shape[1])
            PLV.append(plv)
            
        trigger = np.unique(Trigger)
        PLV_mean = []
    
        for tg in trigger:
            count = 0
            sumplv = 0
        
            for trialINX, tritg in enumerate(Trigger):
                
                if tritg == tg:
                    count = count + 1
                    sumplv = sumplv + PLV[trialINX]
            PLV_mean.append(sumplv/count)
            
        fig,ax = plt.subplots()
        plt.plot(frequency, PLV_mean, marker='o', color='k')
        plt.xlabel('frequency(Hz)')
        plt.ylabel('phase locking value')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    else:
        print('Invalid input\n')

def SPlotTFR(eeg, Trigger, frequency, intensity, alpha, freq, intens=1, trindex=0, nperseg=2000, nfft=5000, fs=1000, vmax=None, uplim=20):
    
    intensity = np.array([int(ins*10) for ins in intensity])
    Findex = np.where(frequency==freq)[0][0]
    Iindex = np.where(intensity==int(intens*10))[0][0]
    tgindex = Findex*len(intensity)+Iindex
    
    eeg_pick = []
    
    for trialINX, tg in enumerate(Trigger):
        if tg == tgindex+1:
             eeg_pick.append(trialINX)
             
    fig,ax = plt.subplots(figsize=(7,5))
    f, t, Sxx = signal.spectrogram(eeg[eeg_pick[trindex],:], fs, nperseg=nperseg, nfft=nfft, noverlap=int(nperseg/2))
    if vmax == None:
        plt.pcolormesh(t, f, Sxx, cmap='Reds', shading='gouraud')
    else:
        plt.pcolormesh(t, f, Sxx, cmap='Reds', shading='gouraud', vmax=vmax)
    plt.hlines(y=alpha, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles='dashed', colors='k')
    plt.hlines(y=freq, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles='dashed', colors='b')
    plt.colorbar()
    plt.ylim(2,uplim)

#################################################################################################   
# Ongoing power methods
################################################################################################# 
def OngoingP(eeg, Trigger, nperseg=5000, nfft=5000, fs=1000):
    
    trigger = np.unique(Trigger)
    eeg_welch = []
    
    for tg in trigger:
        count = 0
        Pxxs = 0
        
        for trialINX, tritg in enumerate(Trigger):
            
            if tritg == tg:
                count = count + 1
                f, Pxx = signal.welch(eeg[trialINX,:], fs, nperseg=nperseg, nfft=nfft)
                Pxxs = Pxxs + Pxx
        eeg_welch.append(Pxxs/count)
    
    return f, eeg_welch

def OPlotspectrum(f, eeg_welch, frequency, intensity, alpha, freq, intens, all=False):
    
    if all == False:
        intensity = np.array([int(ins*10) for ins in intensity])
        Findex = np.where(frequency==freq)[0][0]
        Iindex = np.where(intensity==int(intens*10))[0][0]
        tgindex = Findex*len(intensity)+Iindex
        
        fig,ax = plt.subplots()
        plt.plot(f, eeg_welch[tgindex])
        plt.xlim(0,40)  
        plt.show()  
        
    else:
        trigger = np.arange(len(frequency)*len(intensity))
        stim_FI = np.zeros((len(trigger),2))
        
        for tg in trigger:
            stim_FI[tg, 0] = frequency[math.ceil((tg+1)/len(intensity))-1];
            stim_FI[tg, 1] = intensity[tg-math.ceil((tg+1)/len(intensity))*len(intensity)+len(intensity)];
            
        if len(trigger) == 25:
            fig,axes = plt.subplots(figsize=(10,10), nrows=5, ncols=5)
        else:
            fig,axes = plt.subplots(figsize=(2*len(frequency),2*len(intensity)), nrows=len(intensity), ncols=len(frequency))
            
        for (ax,tg) in zip(axes.flatten(),trigger):
            ax.plot(f, eeg_welch[tg], label=str(format(stim_FI[tg,0],'.1f'))+','+str(format(stim_FI[tg,1],'.1f')))
            ax.legend()
            ax.set_xlim(0,40)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.vlines(x=alpha, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='dashed', colors='k')
            ax.vlines(x=stim_FI[tg,0], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], linestyles='dashed', colors='r')
        plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, hspace=0.2, wspace=0.2)
        plt.show()
        
def OPlotHotmap(f, eeg_welch, frequency, intensity, intens=1, vmax=None, uplim=20):
    
    fig,ax = plt.subplots(figsize=(7,5))
    intensity = np.array([int(ins*10) for ins in intensity])
    dataplot = []
    
    for freq in frequency:
        Findex = np.where(frequency==freq)[0][0]
        Iindex = np.where(intensity==int(intens*10))[0][0]
        tgindex = Findex*len(intensity)+Iindex
        dataplot.append(eeg_welch[tgindex])
    dataplot = np.stack(dataplot)
    dataplot = dataplot.T
    
    if vmax == None:
        plt.pcolormesh(frequency, f, dataplot, shading='gouraud')
    else:
        plt.pcolormesh(frequency, f, dataplot, shading='gouraud', vmax=vmax)
    plt.ylim(0,uplim)
    x = range(int(frequency[0]),int(frequency[len(frequency)-1]),1)
    y = range(0,uplim,1)
    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('Stimulation Frequency\Hz')
    plt.ylabel('Response Frequency\Hz')
    plt.colorbar()
    plt.show()
    
def OPlotTFR(eeg, Trigger, frequency, intensity, alpha, freq, intens=1, nperseg=1000, nfft=5000, fs=1000, vmax=None, uplim=20):
    
    trigger = np.unique(Trigger)
    eeg_spectrogram = []
    
    for tg in trigger:
        count = 0
        Sxxs = 0
        
        for trialINX, tritg in enumerate(Trigger):
            
            if tritg == tg:
                count = count + 1
                f, t, Sxx = signal.spectrogram(eeg[trialINX,:], fs, nperseg=nperseg, nfft=nfft)
                Sxxs = Sxxs + Sxx
        eeg_spectrogram.append(Sxxs/count)
    
    intensity = np.array([int(ins*10) for ins in intensity])
    Findex = np.where(frequency==freq)[0][0]
    Iindex = np.where(intensity==int(intens*10))[0][0]
    tgindex = Findex*len(intensity)+Iindex
    
    fig,ax = plt.subplots()
    if vmax == None:
        plt.pcolormesh(t, f, eeg_spectrogram[tgindex], shading='gouraud', cmap='Reds')
    else:
        plt.pcolormesh(t, f, eeg_spectrogram[tgindex], shading='gouraud', cmap='Reds', vmax=vmax)
    plt.hlines(y=alpha, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles='dashed', colors='k')
    plt.hlines(y=freq, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles='dashed', colors='b')
    plt.colorbar()
    plt.ylim(2,uplim)
    plt.show()
    
def OPlotpeakratio(f, eeg_welch, frequency, intensity, alpha, intens=1):
    
    data_alpha = []
    intensity = np.array([int(ins*10) for ins in intensity])
    fig,ax = plt.subplots()

    for freq in frequency:
        Findex = np.where(frequency==freq)[0][0]
        Iindex = np.where(intensity==int(intens*10))[0][0]
        tgindex = Findex*len(intensity)+Iindex
        Pxx = eeg_welch[tgindex]
        count = 0
        sum_alpha = 0
        
        for findex, f_alpha in enumerate(f):
            if f_alpha<= alpha+0.4 and f_alpha>= alpha-0.4:
                sum_alpha = sum_alpha + Pxx[findex]/Pxx.mean()
                count = count + 1
        data_alpha.append(sum_alpha/count)
        
    data_alpha = np.stack(data_alpha)
    data_alpha = data_alpha/data_alpha.mean()
    frequency = frequency/alpha
    
    plt.plot(frequency, data_alpha, marker='o', color='k')
    plt.xlabel('Stimulation frequency in multiples of alpha')
    plt.ylabel('Alpha peak ratio')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.hlines(y=1, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dashed', color='k')
    
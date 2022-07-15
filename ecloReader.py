
import numpy as np
import mne
import os
import sys
import pickle
from scipy import signal
import scipy.io as scio
from tqdm import tqdm

class ecloReader():
    
    def __init__(self, filename, tstart=0, tend=5, srate=1000, iffilter=True, Wn=[2, 40], trilen=120) -> None:
        # file address
        self.filename = filename
        self.subjects = []
        
        
        # parameters
        self.srate = srate
        self.tstart = tstart
        self.tend = tend
        self.iffilter = iffilter
        self.Wn = Wn
        self.trilen = trilen
        
        self.picklename = self.filename.strip(self.filename.split(os.sep)[-1]) + 'datasets'
        if os.path.exists(self.picklename) is False:
            os.makedirs(self.picklename)
        
        mne.set_log_level(verbose='ERROR')
    
        ahList = os.listdir(self.picklename)
        self.alreadyHave = {ah:len(os.listdir(os.path.join(self.picklename, ah))) for ah in ahList}
    
    def readRaw(self):
        # get subject paths
        self.getSubject()
        
        for subpath in tqdm(self.subListpath):
            # for subject level
            subName = subpath.split(os.sep)[-1]
            subpicklepath = os.path.join(self.picklename,subName)
            if os.path.exists(subpicklepath) == False:
                os.makedirs(subpicklepath)
            
            dateList = os.listdir(subpath)
            for dateName in dateList:
                datepath = subpath+os.sep+dateName
                getList = os.listdir(datepath)
                
                for getName in getList:
                    
                    if os.path.splitext(getName)[-1] == '.cnt':
                        getpath = os.path.join(datepath, getName)
                        # read raw data
                        raw = self.getRaw(getpath)
                        # split raw into epoch
                        epoch = self.getEpoch(raw)
                if epoch != []:
                    with open('%s\%s.pickle' % (subpicklepath, dateName), "wb+") as fp:
                        pickle.dump(epoch, fp, protocol=pickle.HIGHEST_PROTOCOL)
        return
    
    def getRaw(self, getpath):
        
        raw = mne.io.read_raw_cnt(
            input_fname = getpath,
            data_format = 'auto',
            preload = True,
            date_format = 'mm/dd/yy')
        
        return raw
    
    def getEpoch(self, raw):
        task_event, task_dict = self.getEvent(raw)
        taskEpoch = mne.Epochs(raw, task_event, event_id=task_dict, 
                               tmin=self.tstart, tmax=self.tend, preload=True, baseline=None)
        # downsample
        taskEpoch.resample(self.srate)
        # filter here
        if self.iffilter == True:
            taskEpoch.filter(self.Wn[0], self.Wn[1])
        
        # get data
        X = taskEpoch.get_data()[:,:-1]
        filteredX = np.zeros(X.shape)
        for trialINX, trial in enumerate(X):
            for channelINX, channel in enumerate(trial):
                # De baseline
                channel = channel - channel.mean()
                filteredX[trialINX, channelINX] = channel
        
        y = []
        newdict = {v: k for k, v in task_dict.items()}
        for event in task_event[:,-1]:
            y.append(int(newdict.get(event)))
        y = np.array(y)
        
        Channels = taskEpoch.ch_names[:-1]
        
        # data dict
        data = dict(
            X = filteredX,
            y = y,
            channel = Channels)
        
        return data
        
    def getEvent(self, raw):
        
        events, event_dict = mne.events_from_annotations(raw)
        # event mistake must be settled
        valid_dict = {k: v for k, v in event_dict.items() if int(k) < 255}
        valid_event =  np.stack([e for e in events if e[-1] in [*valid_dict.values()]])
        x = np.squeeze(raw['Trigger'][0])
        # correct index
        onset = np.squeeze(np.argwhere(np.diff(x) > 0))
        valid_event[:, 0] = onset[:len(valid_event)]
        # SSVEP刺激事件 Epoch
        task_dict = {k: v for k, v in valid_dict.items() if int(k) <= self.trilen}
        task_event = []
        for e in valid_event:
            if e[-1] in task_dict.values():
                task_event.append(e)
        task_event = np.stack(task_event)  
        
        return task_event, task_dict 
            
            
            
    def getSubject(self):
        
        subList = os.listdir(self.filename)
        for sub, times in self.alreadyHave.items():
            if len(os.listdir(os.path.join(self.filename,sub))) == times:
                subList.remove(sub)
        self.subListpath = [os.path.join(self.filename, subName) for subName in subList]
        
if __name__ == '__main__':
        
    filename = os.path.join(os.getcwd(),'codes','eyeclosedSSVEP','experimentdata')
        
    pickleMaker = ecloReader(filename=filename, tstart=-2)
    pickleMaker.readRaw()
        
import numpy as np
import mne
import os
import pickle
import sys
from scipy import signal
import scipy.io as scio
from tqdm import tqdm
from autoreject import AutoReject
import matplotlib.pyplot as plt

class cntReader():
    """
    cntReader has one permission:load .cnt file into pkls
    should be in the heriachy as :
    -exp
        - subject
            - date
                - session
    """

    def __init__(self, fileAdd, stiLEN,srate=250,tstart=0) -> None:
        # file address
        self.fileAdd = fileAdd
        self.subjects = []
        self.sessions = []
        self.srate=srate

        # parameters
        self.stiLEN = stiLEN
        self.tstart = tstart
        self.cwd = sys.path[0]

        exp = self.fileAdd.split(os.sep)[-1]
        self.pickleAdd = os.path.join('curry', exp)
        if os.path.exists(self.pickleAdd) is False:
            os.makedirs(self.pickleAdd)
            
        self.alreadyHave = os.listdir(self.pickleAdd)

    def readRaw(self):
        # get subject name
        self.getSubject()

        for subINX, sub in enumerate(self.subList):
            # for subject level
            self.subName = sub.split('/')[-1]

            folderList = os.listdir(sub)
            if '.DS_Store' in folderList:
                folderList.remove('.DS_Store')

            for datesRecording in folderList:
                # for dates
                folder = sub+os.sep+datesRecording
                fileList = os.listdir(folder)

                for sessionName in fileList:

                    if os.path.splitext(sessionName)[-1] == '.cnt':
                        sessionName = os.path.join(folder, sessionName)
                        # read continous raw data
                        raw = self._getSession(sessionName)
                        # split continous data into epoch
                        epoch = self.epochSplit(raw)

                        self.sessions.append(epoch)

            if self.sessions != []:
                sessions = np.concatenate(self.sessions)
                with open('%s/%s.pickle' % (self.pickleAdd, self.subName), "wb+") as fp:
                    pickle.dump(sessions, fp, protocol=pickle.HIGHEST_PROTOCOL)

                self.sessions = []

        return

    def _getSession(self, sessionName):

        raw = mne.io.read_raw_cnt(
            input_fname=sessionName,
            data_format='auto',
            preload=True,
            date_format='mm/dd/yy')
        
        return raw

    def epochSplit(self, raw):

        #从这里把区分开三种情况区分开
        Events = self.defineEvent(raw)
        DATA = []

        # notch and filter
        raw.notch_filter((50,100))
        
        for this_exp_events in  Events:

            tag, events, events_dict = this_exp_events
            # 睁闭眼30s
            winLEN = self.stiLEN
            taskEpoch = mne.Epochs(raw, events, event_id=events_dict,
                                   tmin=self.tstart, tmax=winLEN+self.tstart, preload=True, baseline=None)
            taskEpoch.drop_channels(['Trigger'])
            # downsample
            taskEpoch.resample(self.srate)
            taskEpoch._filename = raw.filenames[0]+os.sep+tag

            X = taskEpoch.get_data()
            chn = taskEpoch.ch_names

            y= []
            newd = {v: k for k, v in events_dict.items()}
            for event in events:
                y.append(int(newd.get(event[-1])))
            DATA.append((tag,X,y,chn))

        return DATA

    def correctEvent(self,raw):

        events, event_dict = mne.events_from_annotations(raw)
        # 255是结束trigger,去掉结束trigger
        valid_dict = {k: v for k, v in event_dict.items() if int(k) < 255}
        valid_event =  np.stack([e for e in events if e[-1] in [*valid_dict.values()]])

        x = raw.get_data()[-1]
        onset = np.squeeze(np.argwhere(np.diff(x)>0))

        valid_event[:, 0] = onset[:len(valid_event)]

        # 修正了trigger 位置之后只取任务event
        valid_dict = {k: v for k, v in valid_dict.items() if int(k) <= 120}
        valid_event =  np.stack([e for e in valid_event if e[-1] in [*valid_dict.values()]])

        return valid_dict, valid_event

    def defineEvent(self, raw):
        valid_dict, valid_event = self.correctEvent(raw)

        # 把符合条件的event取出来
        Events = []
        
        ssevp_dict = {k: v for k, v in valid_dict.items() if int(k) <= 120}
        task_events =  [e for e in valid_event if e[-1] in [*ssevp_dict.values()]]
        Events.append(('ssvep', task_events, ssevp_dict))

        return Events

    def getSubject(self):

        subList = os.listdir(self.fileAdd)

        for ah in self.alreadyHave:
            subList.remove(ah.split('.')[0])

        if '.DS_Store' in subList:
            subList.remove('.DS_Store')

        self.subList = [os.path.join(self.fileAdd, subName)
                        for subName in subList]

class datasetMaker():
    
    def __init__(self, exp='exp-1',winLEN=2,srate=250,curryAdd = 'curry',tstart=0):

        self.exp = exp
        self.srate=srate
        self.winLEN = winLEN
        self.sampleLEN = round(self.srate*self.winLEN)
        self.tstart = tstart
        cwd = sys.path[0]
        self.rawAdd = os.path.join(cwd, 'raw',self.exp)
        self.curryAdd = os.path.join(cwd, curryAdd,self.exp)
        self.saveAdd = os.path.join(cwd, 'datasets')
        self.stiAdd = os.path.join(cwd,'stimulation',self.exp)

    def readCurry(self):
        loader = cntReader(self.rawAdd,stiLEN=self.winLEN,srate=self.srate,tstart=self.tstart)
        loader.readRaw()

    def readSTI(self,fileName):

        path = self.stiAdd + os.sep + fileName
        STI = scio.loadmat(path)['WN'].T
        return STI


    def ensemble(self):
        # readData
        self.readCurry()

        dataList = os.listdir(path=self.curryAdd)
        if '.DS_Store' in dataList:
            dataList.remove('.DS_Store')
        wholeSet = []
        dataList = sorted(dataList)

        for filename in tqdm(dataList):
            
            subName = filename.split('.')[0]

            # read stimulation
            path = os.path.join(self.curryAdd, filename)
            with open(path, "rb") as fp:
                sessions = pickle.load(fp)
            ssvep = []
            for session in sessions:
                tag,X,y,chnNames = session
                if tag == 'ssvep':
                    ssvep.append(session)


            ssvep = [np.concatenate([data[1] for data in ssvep],axis=0),
                      np.hstack([data[2] for data in ssvep])]

            sub = dict(
                ssvep = ssvep,
                channel = chnNames,
                name = subName,
            )
            # # 记录来自哪个数据集
            wholeSet.append(sub)

        if not os.path.exists(self.saveAdd):
            os.makedirs(self.saveAdd)
        datasetName = os.path.join(self.saveAdd, '%s.pickle'%self.exp)
        with open(datasetName, "wb+") as fp:
            pickle.dump(wholeSet, fp, protocol=pickle.HIGHEST_PROTOCOL)

        return 

if __name__ == '__main__':

    exp = 'exp-1'
    winLEN = 2
    srate = 500
    
    curryMaker = datasetMaker(exp=exp, winLEN=winLEN,
                              srate=srate)
    curryMaker.ensemble()

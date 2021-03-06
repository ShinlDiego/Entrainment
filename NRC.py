import numpy as np
import pandas as pd
from mne.decoding import ReceptiveField
import os
# from eelbrain import *
import pickle
from scipy import stats,signal


class NRC(ReceptiveField):

    def __init__(self, srate, tmin=-0.5, tmax=1, alpha=0.9, fill_mean=True) -> None:

        self.tmin=tmin
        self.tmax=tmax
        self.srate=srate
        self.estimator = None
        self.alpha = alpha
        self.fill_mean = fill_mean
        # 'boosting' or 'NRC'
        self.mode = 'boosting'

        pass

    def fit(self,R,S):
        
        epochNUM,chnNUM,_ = R.shape
        laggedLEN = len(self._times_to_delays(self.tmin,self.tmax,self.srate))
        Kernel = np.zeros((epochNUM,chnNUM,laggedLEN))
        Cov_sr = np.zeros((epochNUM,chnNUM,laggedLEN))

        for epochINX,(epoch,sti) in enumerate(zip(R,S)):
            sti = sti[:,np.newaxis]
            laggedS = self._delay_time_series(sti, self.tmin, self.tmax,self.srate,fill_mean=self.fill_mean).squeeze()

            # stimulation whitening
            Cov_ss = laggedS.T.dot(laggedS)
            u,sigma,v = np.linalg.svd(Cov_ss.T.dot(Cov_ss))
            for i in range(len(sigma)):
                if sum(sigma[0:len(sigma)-i]/sum(sigma)) < self.alpha:
                    sigma = 1/sigma
                    sigma[len(sigma)-i:] = 0
                    break
            sigma_app = np.diag(sigma)
            inv_C = u.dot(sigma_app).dot(v)
            
            for chnINX,chn in enumerate(epoch):
                
                Cov_sr[epochINX, chnINX] = (laggedS.T).dot(chn)
                Kernel[epochINX, chnINX] = (inv_C).dot(laggedS.T).dot(chn)

        self.kernel = stats.zscore(Kernel,axis=-1)
        self.Csr = stats.zscore(Cov_sr,axis=-1)

        return self

    def fitBoosting(self,R,S,y):


        epochNUM,chnNUM,n_times = R.shape
        laggedLEN = len(self._times_to_delays(self.tmin,self.tmax,self.srate))
        Kernel = []
        tstep = 1/self.srate

        time = UTS(0, tstep, n_times)
        sensor = Sensor.from_montage('biosemi128')[:chnNUM]

        # for epochINX,(epoch,sti) in enumerate(zip(R,S)):

        #     epoch = NDVar(epoch, (sensor,time))
        #     sti = NDVar(sti, (time))
        #     trf = boosting(y=sti,x=epoch,tstart=-0.5,tstop=2,basis=0.1,partitions=2)
        #     Kernel.append(np.array(trf.h_scaled))

        for label in np.unique(y):

            epoch = R[y==label]
            sti = S[y==label]

            epoch = NDVar(epoch, (Case,sensor,time))
            sti = NDVar(sti, (Case,time))
            trf = boosting(y=sti,x=epoch,tstart=self.tmin,tstop=self.tmax)
            Kernel.append(np.array(trf.h_scaled))

        self.kernel = np.stack(Kernel)

        return self

    def _delay_time_series(self,X, tmin, tmax, sfreq, fill_mean=False):
        """Return a time-lagged input time series.

        Parameters
        ----------
        X : array, shape (n_times[, n_epochs], n_features)
            The time series to delay. Must be 2D or 3D.
        tmin : int | float
            The starting lag.
        tmax : int | float
            The ending lag.
            Must be >= tmin.
        sfreq : int | float
            The sampling frequency of the series. Defaults to 1.0.
        fill_mean : bool
            If True, the fill value will be the mean along the time dimension
            of the feature, and each cropped and delayed segment of data
            will be shifted to have the same mean value (ensuring that mean
            subtraction works properly). If False, the fill value will be zero.

        Returns
        -------
        delayed : array, shape(n_times[, n_epochs][, n_features], n_delays)
            The delayed data. It has the same shape as X, with an extra dimension
            appended to the end.

        Examples
        --------
        >>> tmin, tmax = -0.1, 0.2
        >>> sfreq = 10.
        >>> x = np.arange(1, 6)
        >>> x_del = _delay_time_series(x, tmin, tmax, sfreq)
        >>> print(x_del)  # doctest:+SKIP
        [[2. 1. 0. 0.]
        [3. 2. 1. 0.]
        [4. 3. 2. 1.]
        [5. 4. 3. 2.]
        [0. 5. 4. 3.]]
        """
        delays = self._times_to_delays(tmin, tmax, sfreq)
        # Iterate through indices and append
        delayed = np.zeros(X.shape + (len(delays),))
        if fill_mean:
            mean_value = X.mean(axis=0)
            if X.ndim == 3:
                mean_value = np.mean(mean_value, axis=0)
            delayed[:] = mean_value[:, np.newaxis]
        for ii, ix_delay in enumerate(delays):
            # Create zeros to populate w/ delays
            if ix_delay < 0:
                out = delayed[:ix_delay, ..., ii]
                use_X = X[-ix_delay:]
            elif ix_delay > 0:
                out = delayed[ix_delay:, ..., ii]
                use_X = X[:-ix_delay]
            else:  # == 0
                out = delayed[..., ii]
                use_X = X
            out[:] = use_X
            if fill_mean:
                out[:] += (mean_value - use_X.mean(axis=0))
        return delayed


    def _times_to_delays(self,tmin, tmax, sfreq):
        """Convert a tmin/tmax in seconds to delays."""
        # Convert seconds to samples
        delays = np.arange(int(np.round(tmin * sfreq)),
                        int(np.round(tmax * sfreq) + 1))
        return delays

class RegularizedRF(NRC):
    def __init__(self, srate, tmin=-0.5, tmax=1, alpha=1e4,mTRF=False) -> None:
        self.mTRF = mTRF
        super().__init__(srate, tmin, tmax, alpha)

    def fit(self, R, S):

        # normlization
        # R, S = stats.zscore(R, axis=-1), stats.zscore(S, axis=-1)

        Kernel = []

        for  (epoch, sti) in zip(R, S):
            
            model = ReceptiveField(tmin=self.tmin, tmax=self.tmax, sfreq=self.srate,edge_correction=True,estimator=self.alpha)

            r = epoch[:,np.newaxis,:].T
            s = sti[np.newaxis,np.newaxis,:]

            if self.mTRF is True:
                s = self.filterbank(s).T
            else:
                s = s.T

            model.fit(s,r)
            k = stats.zscore(model.coef_,axis=-1)
            Kernel.append(k)

        self.kernel = np.stack(Kernel)
        return self

    def filterbank(self, S):

        passband = [4,6,8,10,12,14,16,20,30,40,50]
        # passband = np.arange(5,50,5)
        stopband = [6,8,10,12,14,16,18,30,40,50,60]

        n_rate = self.srate//2
        fbNUM = len(passband)

        fbS = np.zeros_like(S)
        fbS = np.tile(fbS,(fbNUM,1,1))

        self.band = []
        for fbINX,(passFre,stopFre) in enumerate(zip(passband,stopband)):

            Wp = [passFre/n_rate, stopFre/n_rate]
            Ws = [(passFre-1)/n_rate, (stopFre+1)/n_rate]

            [N, Wn] = signal.cheb1ord(Wp, Ws, 3, 40)
            [B, A] = signal.cheby1(N, 0.5, Wn, 'bandpass')

            for epochINX, epoch in enumerate(S):
                    fbS[fbINX,epochINX] = signal.filtfilt(B, A, epoch,axis=-1)
            self.band.append(Wp)
        return fbS


    def fitByBatch(self,R,S):
        # ??????????????????Epoch????????????????????????Epoch???

        model = ReceptiveField(tmin=self.tmin, tmax=self.tmax, sfreq=self.srate,edge_correction=True,estimator=self.alpha)

        R = np.transpose(R,(-1,0,1))
        S = S[np.newaxis,:,:].T
        model.fit(S,R)
        self.model = model

        return self
    
    def predict(self, S):

        S = S[np.newaxis, :, :].T
        # reconsructed response
        R_r = self.model.predict(S)
        R_r = np.transpose(R_r,(1,-1,0))
        
        return R_r

    def score(self,S,R):

        S = S[np.newaxis, :, :].T
        R = np.transpose(R,(-1,0,1))
        # reconsructed response

        score = self.model.score(S,R)

        return score
stats
        
class recordModule():

    def __init__(self,sub='sub1',recordAdd='results',exp='exp-1',srate=240,chn=np.linspace(0,63,64).astype('int64')):

        self.subName = str(sub)
        self.recordAdd=recordAdd
        self.exp = exp
        self.chnMontage = chn
        self.srate=srate
        self.checkFolder()


    def checkFolder(self):

        folder = os.path.join(self.recordAdd, self.exp,self.subName)
        if os.path.exists(folder) is False:
            os.makedirs(folder)
        self.recordAdd = folder
        return

    def recordEEG(self,*X):

        if len(X) ==2:
            EEG,labels = X
            remark = 'real'
        elif len(X) ==3:
            EEG,labels,remark = X
        else:
            raise Exception('Input variables must contain data and labels')

        _,chnNUM,T = EEG.shape
        t = np.arange(0,T/self.srate,1/self.srate)

        frames = []
        for epoch,label in zip(EEG,labels):
            
            frame = pd.DataFrame(data=epoch,index=self.chnMontage[:chnNUM],columns=t)
            frame.reset_index(level=0, inplace=True)
            frame = frame.melt(id_vars='index', value_name='EEG',
                               var_name='time')
            frame = frame.rename(columns={'index': 'channel'})
            frame['condition'] = label
            frame['remark'] = remark
            
            frames.append(frame)

        filePath = self.recordAdd+os.sep+'EEG.csv'
        df = pd.concat(frames,ignore_index=True,axis=0)
        df['tag'] = self._addTags(df.condition)
        df['subject'] = self.subName

        if os.path.exists(filePath):
            exsited = pd.read_csv(filePath)
            df = pd.concat([df,exsited],ignore_index=True,axis=0)
            df.drop_duplicates()

        df.to_csv(filePath)

        return 

    def recordStimulus(self,*X):

        S,y = X
        # labels = np.arange(1,len(S)+1,step=1)
        T = S.shape[-1]
        t = np.arange(0, T/self.srate, 1/self.srate)
        frame = pd.DataFrame(data=S,index=y, columns=t)
        frame.reset_index(level=0, inplace=True)

        frame = frame.melt(id_vars='index', value_name='stimulus',
                           var_name='time')
        frame = frame.rename(columns={'index': 'condition'})
        frame['tag'] = self._addTags(frame.condition)

        df = frame
        filePath = self.recordAdd+os.sep+'stimulus.csv'

        if os.path.exists(filePath):
            exsited = pd.read_csv(filePath)
            df = pd.concat([df, exsited])
            df.drop_duplicates()
        df.to_csv(filePath)

        return

    def recordKernel(self,*ensemble):
        
        # kernel epoch*chn*T
        if len(ensemble) ==5:
            kernel, labels, type,tmin,tmax = ensemble
        elif len(ensemble) ==4:
            kernel, labels, tmin,tmax = ensemble
        elif len(ensemble) == 3:
            kernel, labels, type = ensemble
            tmin,tmax = 0,kernel.shape[-1]/self.srate
        elif len(ensemble) == 2:
            kernel,labels ==  ensemble
            type = 'regularized'
            tmin, tmax = 0, kernel.shape[-1]/self.srate
        else:
            raise Exception('Input variables must contain data and labels')

        _,chnNUM,featureNUM,T = kernel.shape
        lags = np.arange(tmin, tmax+(1/self.srate), 1/self.srate)
        frames = []

        for epoch,label in zip(kernel,labels):

            epoch = np.transpose(epoch,axes=(1,0,-1))

            for featureINX,feature in enumerate(epoch):
                frame = pd.DataFrame(data=feature,index=self.chnMontage[:chnNUM], columns=lags)
                frame.reset_index(level=0, inplace=True)
                frame = frame.melt(id_vars='index', value_name='trf',var_name='lags')
                frame = frame.rename(columns={'index': 'channel'})
                frame['condition'] = label
                frame['feauture'] = featureINX

                frames.append(frame)
        
        df = pd.concat(frames, ignore_index=True)
        df['subject'] = self.subName
        df['type'] = type
        df['tag'] = self._addTags(df.condition)

        filePath = self.recordAdd+os.sep+'TRF.csv'
        if os.path.exists(filePath):
            exsited = pd.read_csv(filePath)
            df = pd.concat([df,exsited])
            df.drop_duplicates()

        df.to_csv(filePath)
        return

 

    def recordSpectral(self,*X):
        
        if len(X) ==2:
            EEG,labels = X
            remark = 'real'
        elif len(X) ==4:
            freqz,EEG,labels,remark = X
        else:
            raise Exception('Input variables must contain data and labels')

        _,chnNUM,T = EEG.shape

        frames = []
        for epoch,label in zip(EEG,labels):
            
            frame = pd.DataFrame(data=epoch,index=self.chnMontage[:chnNUM],columns=freqz)
            frame.reset_index(level=0, inplace=True)
            frame = frame.melt(id_vars='index', value_name='amplitude',
                               var_name='frequency')
            frame = frame.rename(columns={'index': 'channel'})
            frame['condition'] = label
            frame['remark'] = remark
            
            frames.append(frame)

        filePath = self.recordAdd+os.sep+'spectral.csv'
        df = pd.concat(frames,ignore_index=True,axis=0)
        df['tag'] = self._addTags(df.condition)
        df['subject'] = self.subName

        if os.path.exists(filePath):
            exsited = pd.read_csv(filePath)
            df = pd.concat([df,exsited],ignore_index=True,axis=0)
            df.dropna(how='all')
            df.drop_duplicates()

        df.to_csv(filePath)

        return

    def _addTags(self,conditionINX):

        conditionNUM = 20
        tagNames = np.repeat(['wn','mseq','ssvep'],conditionNUM)
        tags = [tagNames[i-1] for i in conditionINX.to_numpy()]

        return tags
if __name__ == '__main__':

    # R = np.random.random((10, 64, 30))
    R = np.random.random((10,64,720))
    y = np.arange(1,10,1)
    S = np.random.random((10, 720))

    # dir = './datasets/exp-2.pickle'
    # with open(dir, "rb") as fp:
    #     wholeset = pickle.load(fp)
    # sub = wholeset[5]
    # chnNames = ['CPZ', 'PZ', 'POZ', 'OZ', 'P1', 'P2', 'P3']
    # chnINX = [sub['channel'].index(i) for i in chnNames]

    # R = sub['X'][:,chnINX]
    # S = sub['stimulus'].astype('float64')
    # y = sub['y']

    rf = RegularizedRF(srate=240,mTRF=False)
    rf.fitByBatch(R, S)
    rf.predict(S)

    # recoder = recordModule()
    # recoder.recordKernel(rf.kernel,y,'r',)

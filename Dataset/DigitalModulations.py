__author__ = "Brian Shevitski"
__email__ = "brian.shevitski@gmail.com"

import numpy as np
from Dataset.ModulationTypes import GetModulation
from Dataset.FilterUtils import RRC_filt,RRC_filt_RealSpace
from torch.utils.data import Dataset
from scipy.signal import upfirdn
import torch,numba
    
class DigitalModulationDataset(Dataset):
    '''
        Generates a dataset of random symbols encoded using 13 different digital modulations.
        -------------------------------------------------------------------------
        num_examples: how many examples of each modulation to include in data set.
        data_rate: frequency in Hz of the IQ symbol keying. Default is data rate as base unit. 
        num_samples: length of final output data vector.
        seed: random seed value for reproducibility if needed. 
        transform: option to apply pytorch data transforms for channel model.
        normalize_transform: additional pytorch data transforms for normalization.
        min_samp/max_samp: range of values to use for oversampling of each IQ symbol.
        min_RRC_alpha/max_RRC_alpha: range of values to use for RRC filter excess bandwidth.
        classes: if 'all' uses all modulation classes, else takes a list of subset of classes.
        need_tx: makes an extra copy of data that is not transformed by channel.
        return_message: if True returns the symbols encoded in each message, will not work in dataloader with variable message length.
        '''
    def __init__(self, num_examples, data_rate=1, num_samples=512, seed=None, transform=None, normalize_transform=None,
                 min_samp=8,max_samp=20, min_RRC_alpha=0.1, max_RRC_alpha=0.4, classes='all',
                 need_tx=False, return_message=False):
        
        self.num_examples = num_examples # number of examples for each modulation type
        self.data_rate = data_rate #frequency in Hz on the IQ symbol keying. 
                                   #Default is to use data rate as base unit.
        self.num_samples = num_samples # length of final output
        self.seed = seed #random seed for reproducibility
        if self.seed != None:
            self.rng = np.random.default_rng(self.seed) #new way to do numpy random
        else:
            self.rng = np.random.default_rng()
        self.transform = transform #pytorch transform compose
        self.normalize_transform = normalize_transform #pytorch transform compose for normalization only, applied to Tx also
        self.min_samp = min_samp #num_symbols lower bound
        self.max_samp = max_samp #num_symbols upper bound
        self.min_RRC_alpha = min_RRC_alpha #num_symbols lower bound
        self.max_RRC_alpha = max_RRC_alpha #num_symbols upper bound
        self.need_tx = need_tx #returns Tx copy of data
        self.return_message = return_message
        self.rrc_n_taps = num_samples/2 #Eh, Maybe?
        self.rrc_n_taps = numba.int32(self.rrc_n_taps - ((self.rrc_n_taps-1) % 2)) #odd
        
        if classes == 'all':
            self.classes = np.array(['OOK','4ASK','8ASK','BPSK','QPSK','Pi4QPSK','8PSK','16PSK','16QAM','32QAM','64QAM','16APSK','32APSK'])
            self.class_labels = np.arange(0,len(self.classes),1)
            
        else:
            self.classes = np.array(classes)
            self.class_labels = np.arange(0,len(self.classes),1)
    
        self.len = len(self.classes) * self.num_examples
        self.data = []
        self.messages = []
            
        for c in self.classes: # modulations loop
            for _ in range(self.num_examples): # examples loop
                # oversampling and RRC alpha are randomly varied in each example.
                
                #getting the number of message symbols from oversampling parameter
                oversamp = self.rng.integers(self.min_samp,self.max_samp,endpoint=True) 
                num_symbols = self.num_samples/oversamp
                num_symbols = int(np.ceil(num_symbols)) #one extra partial symbol at end 
                      
                #getting RRC alpha
                alpha = self.rng.uniform(low=self.min_RRC_alpha,high=self.max_RRC_alpha)
                
                if self.return_message:
                #makes random symbol strings, (see GetModulation in ModulationTypes.py)
                    if self.seed != None:
                        (symbols_I,symbols_Q), symbols = GetModulation(c,num_symbols,self.rng)
                    else:
                        (symbols_I,symbols_Q), symbols = GetModulation(c,num_symbols)
                        
                else:
                #makes random symbol strings, (see GetModulation in ModulationTypes.py)
                    if self.seed != None:
                        (symbols_I,symbols_Q), _ = GetModulation(c,num_symbols,self.rng)
                    else:
                        (symbols_I,symbols_Q), _ = GetModulation(c,num_symbols) 
                
                #applying sampling to symbols
                digital_I = upfirdn(np.ones(oversamp),symbols_I,up=oversamp) 
                digital_Q = upfirdn(np.ones(oversamp),symbols_Q,up=oversamp)
                
                #RRC pulse shaping
                dt = 1/float(self.data_rate*oversamp)
                
                #digital_I = RRC_filt(digital_I,dt,alpha,1/self.data_rate)
                #digital_Q = RRC_filt(digital_Q,dt,alpha,1/self.data_rate)
                #Using real-space version with arbitrary number of taps for now because of wrapping issue.
                digital_I = RRC_filt_RealSpace(digital_I,self.rrc_n_taps,alpha,self.data_rate,1/float(dt))
                digital_Q = RRC_filt_RealSpace(digital_Q,self.rrc_n_taps,alpha,self.data_rate,1/float(dt))
            
                #Does a random offset and resamples data to be proper size
                #timing is basically destroyed
                #print("Length:" + str(len(digital_I)))
                diff_index = len(digital_I)-self.num_samples
                if diff_index != 0:
                    #print("Diff index:" + str(diff_index))
                    random_index = self.rng.integers(0,diff_index)
                    #print("Rand index:" + str(random_index))
                    digital_I = digital_I[random_index:random_index+self.num_samples]
                    digital_Q = digital_Q[random_index:random_index+self.num_samples]
                
                class_label = np.where(self.classes==c)[0][0]
                   
                if self.return_message:
                    self.messages.append(symbols)
                
                self.data.append([class_label,num_symbols,alpha,oversamp,dt,[digital_I.astype('float32'),digital_Q.astype('float32')]])
                
    def __len__(self):
        return self.len

    def __getitem__(self,idx):

        examp = self.data[idx]
        data = torch.tensor(examp[-1])
        label = examp[0]
        meta = {'num_symbols':examp[1],'RRC_alpha':examp[2],'oversamp':examp[3],'dt':examp[4]}
        
        if self.return_message:
            message = self.messages[idx]
            return_dict = {'data' : data,'label' : label,'metadata':meta, 'message':message}
        else:
            return_dict = {'data' : data,'label' : label,'metadata':meta}
        
        if self.need_tx == True:
            return_dict['data_Tx']=torch.clone(return_dict['data'])
        
        if self.transform != None:
            data = self.transform(return_dict)
        
        if self.normalize_transform != None:
            data = self.normalize_transform(return_dict)
        
        #return data
        return return_dict
    
class DigitalDemodulationDataset(Dataset):
    '''
        Generates a dataset of random symbols encoded using 13 different digital modulations.
        -------------------------------------------------------------------------
        num_examples: how many examples of each modulation to include in data set.
        data_rate: frequency in Hz of the IQ symbol keying. Default is data rate as base unit. 
        num_samples: length of final output data vector.
        seed: random seed value for reproducibility.
        transform: option to apply pytorch data transforms for channel model.
        normalize_transform: additional pytorch data transforms for normalization.
        min_samp/max_samp: range of values to use for oversampling of each IQ symbol.
        min_RRC_alpha/max_RRC_alpha: range of values to use for RRC filter excess bandwidth.
        classes: if 'all' uses all modulation classes, else takes a list of subset of classes.
        need_tx: makes a copy of data that is not transformed by channel
        return_message: if True returns the symbols encoded in each message, will not work in dataloader with variable message length.
        gen_samples: if False the oversampling is randomly generated and the number of symbols is calculated, num_symbols = num_samples/oversamp, 
        if True the number of symbols is randomly generated and the oversampling is calculated, oversamp = num_samples/num_symbols. 
        '''
    def __init__(self, num_examples, data_rate=1, num_samples=512, seed=None, transform=None, normalize_transform=None,
                 min_samp=8, max_samp=20, min_RRC_alpha=0.1, max_RRC_alpha=0.4, classes='all',
                 need_tx=False, return_message=True,gen_samples=False):
        
        self.num_examples = num_examples # number of examples for each modulation type
        self.data_rate = data_rate #frequency in Hz on the IQ symbol keying. 
                                   #Default is to use data rate as base unit.
        self.num_samples = num_samples # length of final output
        self.seed = seed #random seed for reproducibility
        if self.seed != None:
            self.rng = np.random.default_rng(self.seed) #new way to do numpy random
        else:
            self.rng = np.random.default_rng()
        self.transform = transform #pytorch transform compose
        self.normalize_transform = normalize_transform #pytorch transform compose for normalization only, applied to Tx also
        self.min_samp = min_samp #num_symbols lower bound
        self.max_samp = max_samp #num_symbols upper bound
        self.min_RRC_alpha = min_RRC_alpha #num_symbols lower bound
        self.max_RRC_alpha = max_RRC_alpha #num_symbols upper bound
        self.need_tx = need_tx #returns Tx copy of data
        self.return_message = return_message
        self.rrc_n_taps = num_samples/2 #Eh, Maybe?
        self.rrc_n_taps = numba.int32(self.rrc_n_taps - ((self.rrc_n_taps-1) % 2)) #odd
        self.gen_samples = gen_samples

        if classes == 'all':
            self.classes = np.array(['OOK','4ASK','8ASK','BPSK','QPSK','Pi4QPSK','8PSK','16PSK','16QAM','32QAM','64QAM','16APSK','32APSK'])
            self.class_labels = np.arange(0,len(self.classes),1)
            
        else:
            self.classes = np.array(classes)
            self.class_labels = np.arange(0,len(self.classes),1)
    
        self.len = len(self.classes) * self.num_examples
        self.data = []
        self.messages = []
            
        for c in self.classes: # modulations loop
            for _ in range(self.num_examples): # examples loop
                # oversampling and RRC alpha are randomly varied in each example.
                
                if self.gen_samples:
                    
                    #getting the oversampling parameter from the number of message symbols
                    num_symbols = self.rng.integers(self.min_samp,self.max_samp,endpoint=True)   
                    oversamp = int(np.ceil(self.num_samples/num_symbols))
                   
                else:
                
                    #getting the number of message symbols from oversampling parameter
                    oversamp = self.rng.integers(self.min_samp,self.max_samp,endpoint=True) 
                    num_symbols = self.num_samples/oversamp
                    num_symbols = int(np.ceil(num_symbols)) #one extra partial symbol at end 
                      
                #getting RRC alpha
                alpha = self.rng.uniform(low=self.min_RRC_alpha,high=self.max_RRC_alpha)
                
                if self.return_message:
                #makes random symbol strings, (see GetModulationAndBitString in ModulationTypes.py)
                    if self.seed != None:
                        (symbols_I,symbols_Q), symbols = GetModulation(c,num_symbols,self.rng)
                    else:
                        (symbols_I,symbols_Q), symbols = GetModulation(c,num_symbols)
                        
                else:
                #makes random symbol strings, (see GetModulation in ModulationTypes.py)
                    if self.seed != None:
                        (symbols_I,symbols_Q),_ = GetModulation(c,num_symbols,self.rng)
                    else:
                        (symbols_I,symbols_Q),_= GetModulation(c,num_symbols) 
                
                #applying sampling to symbols
                digital_I = upfirdn(np.ones(oversamp),symbols_I,up=oversamp) 
                digital_Q = upfirdn(np.ones(oversamp),symbols_Q,up=oversamp)
                
                #RRC pulse shaping
                dt = 1/float(self.data_rate*oversamp)
                #digital_I = RRC_filt(digital_I,dt,alpha,1/self.data_rate)
                #digital_Q = RRC_filt(digital_Q,dt,alpha,1/self.data_rate)
                #Using real-space version with arbitrary number of taps for now because of wrapping issue.
                digital_I = RRC_filt_RealSpace(digital_I,self.rrc_n_taps,alpha,self.data_rate,1/float(dt))
                digital_Q = RRC_filt_RealSpace(digital_Q,self.rrc_n_taps,alpha,self.data_rate,1/float(dt))
            
            
                #Assumes timing is synchronized at t=0 and resamples
                #leaves a partial symbol at the end if oversample numbers are not nice numbers
                digital_I = digital_I[0:self.num_samples]
                digital_Q = digital_Q[0:self.num_samples]
                
                class_label = np.where(self.classes==c)[0][0]
                   
                if self.return_message:
                    self.messages.append(symbols)
                
                self.data.append([class_label,num_symbols,alpha,oversamp,dt,[digital_I.astype('float32'),digital_Q.astype('float32')]])
                
    def __len__(self):
        return self.len

    def __getitem__(self,idx):

        examp = self.data[idx]
        data = torch.tensor(examp[-1])
        label = examp[0]
        meta = {'num_symbols':examp[1],'RRC_alpha':examp[2],'oversamp':examp[3],'dt':examp[4]}
        
        if self.return_message:
            message = self.messages[idx]
            return_dict = {'data' : data,'label' : label,'metadata':meta, 'message':message}
        else:
            return_dict = {'data' : data,'label' : label,'metadata':meta}
            
        if self.need_tx == True:
            return_dict['data_Tx']=torch.clone(return_dict['data'])
        
        if self.transform != None:
            data = self.transform(return_dict)
        
        if self.normalize_transform != None:
            data = self.normalize_transform(return_dict)

        #return data
        return return_dict

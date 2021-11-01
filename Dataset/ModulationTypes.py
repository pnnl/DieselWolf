__author__ = "Brian Shevitski"
__email__ = "brian.shevitski@gmail.com"

#definition of basic digital modulation types
#please note that the mapping between symbols and constellation points is not Gray coded.
#Symbols and constellation points map to one another arbitrarily.

import numpy as np

def GetModulation(modulation,num_symbols,rng=False):

    '''
    modulation: string, name of modulation type to get 
    num_symbols: int, how many symbols to fetch from the constellation for the message
    rng: if False creates a default numpy rng generator object, otherwise takes a np.random.random_generator()
    seed: if False returns random messages, if int returns predetermined messages for reproducability.
    '''
    if rng:
        rng = rng
    else:
        rng = np.random.default_rng()
    
    if modulation == 'BPSK':
        symbols = rng.choice([0,1],num_symbols)
        symbols_I = np.cos(2*np.pi*(symbols)/2)
        symbols_Q = np.sin(2*np.pi*(symbols)/2)

    elif modulation == 'QPSK':
        symbols = rng.choice([0,1,2,3],num_symbols)
        symbols_I = np.cos(2*np.pi*(symbols)/4 + np.pi/4)
        symbols_Q = np.sin(2*np.pi*(symbols)/4 + np.pi/4)
    
    elif modulation == 'OQPSK':#not implemented
        #symbols = rng.choice([0,1,2,3],num_symbols)
        #symbols_I = np.cos(2*np.pi*(symbols)/4 + np.pi/4)
        #symbols_Q = np.sin(2*np.pi*(symbols)/4 + np.pi/4)
        raise NotImplementedError("Sorry, you have to figure this one out yourself :(")

    elif modulation == 'Pi4QPSK':
        symbols = rng.choice([0,1,2,3],num_symbols)
        symbols_I_even = np.cos(2*np.pi*(symbols[::2])/4 + np.pi/4)
        symbols_Q_even = np.sin(2*np.pi*(symbols[::2])/4 + np.pi/4)
        symbols_I_odd = np.cos(2*np.pi*(symbols[1::2])/4)
        symbols_Q_odd = np.sin(2*np.pi*(symbols[1::2])/4)

        symbols_I = np.empty((symbols_I_even.size + symbols_I_odd.size,),dtype=symbols_I_even.dtype)
        symbols_I[0::2] = symbols_I_even
        symbols_I[1::2] = symbols_I_odd

        symbols_Q = np.empty((symbols_Q_even.size + symbols_Q_odd.size,),dtype=symbols_Q_even.dtype)
        symbols_Q[0::2] = symbols_Q_even
        symbols_Q[1::2] = symbols_Q_odd

    elif modulation == '8PSK':
        symbols = rng.choice([0,1,2,3,4,5,6,7],num_symbols)
        symbols_I = np.cos(2*np.pi*(symbols)/8 + np.pi/4)
        symbols_Q = np.sin(2*np.pi*(symbols)/8 + np.pi/4)

    elif modulation == '16PSK':
        symbols = rng.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],num_symbols)
        symbols_I = np.cos(2*np.pi*(symbols)/16 + np.pi/4)
        symbols_Q = np.sin(2*np.pi*(symbols)/16 + np.pi/4)

    elif modulation == 'OOK':
        symbols = rng.choice([0,1],num_symbols)
        symbols_I = symbols
        symbols_Q = symbols*0
        
    elif modulation == '4ASK':
        symbols = rng.choice([0,1,2,3],num_symbols)
        symbols_I = symbols
        symbols_Q = symbols*0
        
    elif modulation == '8ASK':
        symbols = rng.choice([0,1,2,3,4,5,6,7],num_symbols)
        symbols_I = symbols
        symbols_Q = symbols*0
    
    elif modulation == '16QAM':
        symbols_I = rng.choice([-3,-1,1,3],num_symbols)
        symbols_Q = rng.choice([-3,-1,1,3],num_symbols)
        constellation = np.arange(0,16,1)
        constellation = np.reshape(constellation,(4,4))
        constellation = np.insert(constellation,-1,np.zeros(4).astype(int),axis=1)
        constellation = np.insert(constellation,2,np.zeros(4).astype(int),axis=1)
        constellation = np.insert(constellation,1,np.zeros(4).astype(int),axis=1)
        constellation = np.insert(constellation,-1,np.zeros(7).astype(int),axis=0)
        constellation = np.insert(constellation,2,np.zeros(7).astype(int),axis=0)
        constellation = np.insert(constellation,1,np.zeros(7).astype(int),axis=0)
        symbols = np.array([constellation[q-4,i-4] for i,q in zip(symbols_I,symbols_Q)])

    elif modulation == '64QAM':
        symbols_I = rng.choice([-7,-5,-3,-1,1,3,5,7],num_symbols)
        symbols_Q = rng.choice([-7,-5,-3,-1,1,3,5,7],num_symbols)
        constellation = np.arange(0,64,1)
        constellation = np.reshape(constellation,(8,8))
        for column in np.arange(1,15,2):
            constellation = np.insert(constellation,column,np.zeros(8).astype(int),axis=1)
        for row in np.arange(1,15,2):
            constellation = np.insert(constellation,row,np.zeros(15).astype(int),axis=0)
        symbols = np.array([constellation[q-8,i-8] for i,q in zip(symbols_I,symbols_Q)])
            
    elif modulation == '32QAM':
        symbols_I = rng.choice([-5,-3,-1,1,3,5],num_symbols)
        symbols_Q = []
        for si in symbols_I:
            if abs(si) == 5:
                symbols_Q.append(rng.choice([-3,-1,1,3]))
            else:
                symbols_Q.append(rng.choice([-5,-3,-1,1,3,5]))
        symbols_Q = np.array(symbols_Q)
        constellation = np.arange(0,32,1)
        constellation = np.insert(constellation,0,-1)
        constellation = np.insert(constellation,5,-1)
        constellation = np.insert(constellation,-4,-1)
        constellation = np.insert(constellation,35,-1)
        constellation = np.reshape(constellation,(6,6))
        for column in np.arange(1,11,2):
            constellation = np.insert(constellation,column,-np.ones(6).astype(int),axis=1)
        for row in np.arange(1,11,2):
            constellation = np.insert(constellation,row,-np.ones(11).astype(int),axis=0)
        symbols = np.array([constellation[i-6,q-6] for i,q in zip(symbols_I,symbols_Q)])

    elif modulation == '16APSK':
        
        apsk16_rad1 = 1
        apsk16_rad2 = 2.7
        
        symbols_r = rng.choice([apsk16_rad1,apsk16_rad2],num_symbols)
        symbols_theta = []
        symbols_num = []
        symbols = []
        for r in symbols_r:
            if r==apsk16_rad1:
                roll = rng.choice([0,1,2,3])
                symbols_theta.append(roll)
                symbols_num.append(4)
                symbols.append(roll)
            else:
                roll = rng.choice([0,1,2,3,4,5,6,7,8,9,10,11])
                symbols_theta.append(roll)
                symbols_num.append(12)
                symbols.append(roll+4)
        symbols_theta = np.array(symbols_theta)
        symbols_num = np.array(symbols_num)
     
        symbols_I = []
        symbols_Q = []

        for r,t,n in zip(symbols_r,symbols_theta,symbols_num):
            symbols_I.append(r*np.cos(2*np.pi*t/n + (np.pi/4)))
            symbols_Q.append(r*np.sin(2*np.pi*t/n + (np.pi/4)))
        symbols_I = np.array(symbols_I)
        symbols_Q = np.array(symbols_Q)
        symbols = np.array(symbols)

    elif modulation == '32APSK':
        
        apsk32_rad1 = 1
        apsk32_rad2 = 2.64
        apsk32_rad3 = 4.64
        
        symbols_r = rng.choice([apsk32_rad1,apsk32_rad2,apsk32_rad3],num_symbols)
        symbols_theta = []
        symbols_num = []
        symbols = []
        for r in symbols_r:
            if r==apsk32_rad1:
                roll = rng.choice([0,1,2,3])
                symbols_theta.append(roll)
                symbols_num.append(4)
                symbols.append(roll)
            elif r == apsk32_rad2:
                roll = rng.choice([0,1,2,3,4,5,6,7,8,9,10,11])
                symbols_theta.append(roll)
                symbols_num.append(12)
                symbols.append(roll+4)
            else:
                roll = rng.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
                symbols_theta.append(roll)
                symbols_num.append(16)
                symbols.append(roll+16)
        symbols_theta = np.array(symbols_theta)
        symbols_num = np.array(symbols_num)
    
        symbols_I = []
        symbols_Q = []

        for r,t,n in zip(symbols_r,symbols_theta,symbols_num):
            symbols_I.append(r*np.cos(2*np.pi*t/n + (np.pi/4)))
            symbols_Q.append(r*np.sin(2*np.pi*t/n + (np.pi/4)))
        symbols_I = np.array(symbols_I)
        symbols_Q = np.array(symbols_Q)
        symbols = np.array(symbols)

    else:
        raise NameError("Not a valid modulation")
        
    return (symbols_I,symbols_Q),symbols
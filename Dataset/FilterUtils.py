import numpy as np
import numba
from scipy.ndimage.filters import convolve1d

def butter_freq(f,f0,n): #butterworth lowpass frequency response
    g = 1/(1+(f/f0)**(2*n))
    return g

def butter_filt(sig,samp,f0,n): #butterworth lowpass frequency filter
    fft = np.fft.fftshift(np.fft.fft(sig))
    freqs = np.fft.fftshift(np.fft.fftfreq(n=fft.size, d=1/samp))
    masked = butter_freq(freqs,f0,n)*fft
    filt = np.fft.ifft(np.fft.ifftshift(masked))
    return np.real(filt)

def RRC_freq(f,a,t): #root-raised cosine BW filter for pulse shaping
    m1 = abs(f) <= (1-a)/(2*t) #a is excess BW param and t is data symbol time
    m2 = (abs(f) > (1-a)/(2*t) ) * ( abs(f) <= (1+a)/(2*t))
    m3 = np.invert(np.invert(m1)*np.invert(m2))
    
    v1 = 1
    v2 = np.sqrt(0.5*(1+np.cos(np.pi*t/a*(abs(f)-((1-a)/(2*t))))))
    
    return m3*(v1*m1+m2*v2)

def RRC_filt(sig,dt,a,t): #RRC pulse shape filter
    fft = np.fft.fftshift(np.fft.fft(sig))
    freqs = np.fft.fftshift(np.fft.fftfreq(n=fft.size, d=dt))
    masked = RRC_freq(freqs,a,t)*fft
    filt = np.fft.ifft(np.fft.ifftshift(masked))
    return np.real(filt)

@numba.njit #real-space convolution implementation of RRC filter
def rrcosfilter(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.
    Parameters
    ----------
    Adapted from CommPy 
    ----------
    N : int
        Length of the filter in samples.
    alpha : float
        Roll off factor (Valid values are [0, 1]).
    Ts : float
        Symbol period in seconds.
    Fs : float
        Sampling Rate in Hz.
    Returns
    ---------
    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    """

    T_delta = 1/float(Fs)
    time_idx = ((np.arange(N)-N/2))*T_delta
    sample_num = np.arange(N)
    h_rrc = np.zeros(N, dtype=numba.float32)

    for x in sample_num:
        t = (x-N/2)*T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4*alpha/np.pi)
        elif alpha != 0 and t == Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        elif alpha != 0 and t == -Ts/(4*alpha):
            h_rrc[x] = (alpha/np.sqrt(2))*(((1+2/np.pi)* \
                    (np.sin(np.pi/(4*alpha)))) + ((1-2/np.pi)*(np.cos(np.pi/(4*alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi*t*(1-alpha)/Ts) +  \
                    4*alpha*(t/Ts)*np.cos(np.pi*t*(1+alpha)/Ts))/ \
                    (np.pi*t*(1-(4*alpha*t/Ts)*(4*alpha*t/Ts))/Ts)

    return time_idx, h_rrc

def RRC_filt_RealSpace(_array,N_taps,alpha,Ts,Fs,mode='constant'):
    
    kernel = rrcosfilter(N_taps,alpha,Ts,Fs)[1]
    _output = convolve1d(_array,kernel,mode=mode)/kernel.sum()
    
    return _output
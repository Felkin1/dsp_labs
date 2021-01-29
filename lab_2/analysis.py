
from scipy.signal import kaiserord, lfilter, firwin, freqz, impulse2, tf2zpk
from plots import *
import scipy as scipy

def impz(b,a=1,sz=1):
    # generate impulse response of given filter coefs

    zeros = scipy.zeros(sz)
    zeros[0] = 1
    return lfilter(b,a, zeros)

def analyze_filter(b,a,fd,analysis_types=[1,2,3,4,100]):
    # impulse, frequency, phase and zplane analysis of a given filter
    for t in analysis_types:
        if t == 1:
            #1 impz
            if len(analysis_types) > 4:
                impz_h = impz(b,a,analysis_types[4])
            else:
                impz_h = impz(b,a,len(b))
            plot_impulse_response(impz_h)
        elif t == 2:
            #2 freq
            w,h = freqz(b=b,a=a,fs=fd)
            plot_filter_amplitude_frequency(w=w,h=h,normalized=False,filename=None)
        elif t == 3:
            #3 phase
            plot_filter_phase(w,h,normalized=False,filename=None)
        elif t == 4:
            #4 poles and zeros
            plot_zplane(b,a,filename=None)
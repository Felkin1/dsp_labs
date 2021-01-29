from scipy.signal import kaiserord, lfilter, firwin, freqz, impulse2, tf2zpk
import numpy as np
from scipy import signal
from utils import normalize, pad_zeros
import copy as copy
from scipy.ndimage.interpolation import shift
from numpy import cos, sin, pi, absolute, arange

def fir(x,fd,transition_width,attenuation,cutoff):
    #------------------------------------------------
    # Create a FIR filter and apply it to x.
    #------------------------------------------------
    print(f"started a fir filter routine with:\nn={len(x)} fd={fd}")
    print(f"transition_width: {transition_width}hz, attenuation: {attenuation}dB, cutoff: {cutoff}hz")
    # The Nyquist rate of the signal.
    nyq_rate = fd / 2.0

    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.
    width = transition_width/nyq_rate

    # The desired attenuation in the stop band, in dB.
    ripple_db = attenuation

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)

    # The cutoff frequency of the filter.
    cutoff_hz = cutoff

    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    #if analyze:
    #    analysis(taps,[1],fd)
    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, x)
    assert np.all(np.abs(taps - taps[::-1]) < 1e-15)
    delay = int(len(taps) / 2)

    print(f"delay: {delay}")

    return filtered_x, delay,taps

def fir2(x,fd,cutoff,tap_n):
    #------------------------------------------------
    # Create a FIR filter and apply it to x.
    #------------------------------------------------

    taps = firwin(numtaps=tap_n, cutoff=cutoff,window="nuttall",fs=fd)

    # Use lfilter to filter x with the FIR filter.
    filtered_x = lfilter(taps, 1.0, x)

    delay = len(taps) / 2
    #delay = 0
    print(f"delay: {delay}")

    return filtered_x, delay

def fir_lowpass_filter(x,fir_coefs):
    # rir lowpass filter implementation
    #x,mean,multi = normalize(x)

    b = fir_coefs
    a = [1]
    y = lfilter(b,a,x)

    #y,mean,multi = normalize(y,mean=mean,multi=multi,reverse=True)

    return y,a,b

def comb_filter(x,fd,p):
    # comb IIR filter implementation


    #x_mean = x.mean()

    x,mean,multi = normalize(x,mean=1,multi=1,reverse=False)

    # ---------------------------------------------------------------
    # hyperparameters
    # ---------------------------------------------------------------

    S = p["S"]                # slopinimas ties kartotine
    delta_f = p["delta_f"]    # slopinimo plotis ties -3dB
    f0 = p["f0"]              # dazniu kartotine slopinimui
    k0 = p["k0"]              # slopinimas ties pralaidumo juosta (1=jokio)
    L = p["L"]                # filtro pjuvio daznis
    
    # ---------------------------------------------------------------  
    
    n = fd/f0 
    k = k0*(10**(-S/20))    # slopinimas ties kf0 
    kr = k0*(10**(-L/20))   # delta_f vertinimo amplitude (-3dB pagal L)
    print(f"kr: {kr}")
    print(delta_f)
    beta = np.sqrt((kr**2-k0**2)/(k**2-kr**2)) * np.tan((n*pi*delta_f)/(2*fd))

    k1 = (k0+k*beta) / (1+beta)
    k2 = (k0-k*beta) / (1+beta)
    k3 = (1-beta) / (1+beta)

    print(beta,k1,k2,k3)

    b = np.concatenate(([k1] , np.zeros(int(n)-1) , [-k2]),axis=0)
    a = np.concatenate(([1] , np.zeros(int(n)-1) , [-k3]),axis=0)
    print(f"k: {k} n: {n} kr: {kr}")
    print(a)
    print(b)
    y = lfilter(b,a,x)

    #y += x_mean
    y,mean,multi = normalize(y,mean=mean,multi=multi,reverse=True)

    return y,a,b

def decimation(x,fd,factor,transition_width,attenuation,cutoff):

    # fir filtering
    cutoff = fd / (2*factor)
    #x,delay = fir2(x,fd,cutoff,20)
    print(f"decimation cutoff: {cutoff}, fd: {fd}")
    x,delay,taps = fir(x,fd,transition_width,attenuation,cutoff)
    #delay = 0
    print("Shield filtered before decimation downsampling")
    print(f"factor: {factor}")
    # downsampling
    downsampled_size = int(len(x)/factor)
    new_fd = fd / factor
    print(len(x))
    #x = signal.resample(x, downsampled_size)
    #x = signal.resample_poly(x, 1,factor)
    x = x[::factor]
    print(len(x))

    print(f"downsampled to {new_fd} hz")

    return x,new_fd,delay,taps

def interpolation(x,fd,factor,transition_width,attenuation,cutoff):
    #upsampling
    #upsample_factor = int(len(x)*factor)
    new_fd = fd * factor
    #x = signal.resample(x, upsample_factor)
    #x = signal.resample_poly(x, factor, 1)
    x = pad_zeros(x,factor)
    print(f"upsampled to {new_fd} hz")

    # fir filtering
    cutoff = fd / (2)
    x,delay,taps = fir(x,new_fd,transition_width,attenuation,cutoff)
    
    print(f"interp cutoff:{cutoff}, fd: {new_fd}")
    #x,delay = fir2(x,new_fd,cutoff,20)
    #delay = 0
    print("Shield filtered post interpolation upsample")
    #x *= factor*2 #boost
    

    return x,new_fd,delay,taps

def multirate_fir(x,fd,par):
    delay_accumulation = 0

    taps = []  
    
    x,mean,multi = normalize(x)

    M =     50 # issiskaiciuota
    D_1 =   10 # issiskaiciuota
    D_2 =   5  # issiskaiciuota

    fd1 = fd/D_1    # 50
    fd2 = fd1/D_2   # 10

    downscaled_fds =        [fd1, fd2]  # dazniai po pirmos ir antros decim
    factors =               [D_1,D_2]    
    transition_widths  =    [20,5]
    attenuations =          [par["multi_attenuation"], par["multi_attenuation"]]
    cutoffs =               [fd/(2*D_1), fd1/(2*D_2)] # D_1/2 ar M?

    #cutoffs =              [fd/(2*M), fd1/(2*M)]
    #cutoffs =              [0.5, 0.5]
    # ---------------------------------------------------------------



    # saving the original signal for later
    x_buff = copy.copy(x)
    
    x= np.concatenate((x,np.zeros(1500)),axis=0)
    # decimation

    # phase 1
    
    x,fdt,delay,tap = decimation(x,fd,factors[0],
        transition_widths[0],attenuations[0],cutoffs[0])

    #assert fdt == downscaled_fds[0]
    delay_accumulation += delay
    taps.append(tap)

    # phase 2
    
    x,fdt,delay,tap = decimation(x,downscaled_fds[0],factors[1],
        transition_widths[1],attenuations[1],cutoffs[1])

    #assert fdt == downscaled_fds[1]
    delay_accumulation += delay
    taps.append(tap)

    # lowpass filter (zdf, main)
    #plot_signal(x,fd,fd*2,"post decimation")
    x,delay,tap = fir(x,downscaled_fds[1],par["f_transition_width"],
    par["f_attenuation"],par["f_cutoff"])
    delay_accumulation += delay
    taps.append(tap)
    #plot_signal(x,fd,fd*2,"post fir")

    # interpolation

    # phase 1
    x,fdt,delay,tap = interpolation(x,downscaled_fds[1],factors[1],
        transition_widths[1],attenuations[1],cutoffs[1])

    #assert fdt == downscaled_fds[0]
    delay_accumulation += delay
    taps.append(tap)

    # phase 2
    x,fdt,delay,tap = interpolation(x,downscaled_fds[0],factors[0],
        transition_widths[0],attenuations[0],cutoffs[0])

    #assert fdt == fd
    delay_accumulation += delay
    taps.append(tap)

    # delay
    #x_buff = shift(x_buff, delay_accumulation*5, cval=0.0)
    #x = shift(x,-delay_accumulation*5, cval=0.0)
    print(delay_accumulation)
    x *= M
    # combine
    x = x[delay_accumulation*5:delay_accumulation*5+len(x_buff)]
    #x = x[1500:]
    #y = x_buff- x
    #y = x
    y = x_buff - x
    #y = y + abs(y.m)
    

    y,mean,multi = normalize(y,mean=mean,multi=multi,reverse=True)

    #y = x
    # boost amplitude

    return y,x,taps
#analysis.py

from scipy.signal import kaiserord, lfilter, firwin, freqz, impulse2, tf2zpk
from plots import *
import scipy as scipy

def impz(b,a=1):
    # generate impulse response of given filter coefs

    zeros = scipy.zeros(len(b))
    zeros[0] = 1
    return lfilter(b,a, zeros)

def analyze_filter(b,a,fd,analysis_types=[1,2,3,4]):
    # impulse, frequency, phase and zplane analysis of a given filter
    for t in analysis_types:
        if t == 1:
            #1 impz
            impz_h = impz(b,a)
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


#plots.py

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
from utils import *

def plot_test_signal_comparison(x,y,title,filename=None):

    plt.plot(x[0],x[1],'k')
    plt.plot(x[0],y,'--k')
    plt.title(f"{title} test")
    plt.xlabel("t, s")
    plt.ylabel("voltage, V")


    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plot_zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='k')

    ax.set_title("z plane")

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='k', markerfacecolor='k')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    

    return z, p, k

def plot_filter_amplitude_frequency(w,h,normalized=False,filename=None):
    # plot the frequency characteristics

    fig, axs = plt.subplots(1,1,figsize=(8,8),dpi=100)
    #convert to db from amplitude
    h_dB = 20 * np.log10(abs(h))   

    
    if normalized:
        axs.plot(w/np.max(w),h_dB,'k')
        axs.set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    else:
        axs.plot(w,h_dB,'k')
        axs.set_xlabel(r'Frequency, Hz')


    axs.set_ylabel('Magnitude (db)')
    #axs[0].set_ylim(-150, 5)
    axs.grid()
    axs.set_title(r'Frequency response')
    axs.axhline(y=-3.0, color="black", linestyle="--")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plot_filter_phase(w,h,normalized=False,filename=None):
    # plot the phase characteristics

    fig, axs = plt.subplots(1,1,figsize=(8,8),dpi=100)
    #convert to db from amplitude
    h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))

    if normalized:
        axs.plot(w/np.max(w), h_Phase, 'k')
        axs.set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    else:
        axs.plot(w, h_Phase, 'k')
        axs.set_xlabel(r'Frequency, Hz')
    
    axs.set_ylabel('Phase (radians)')
    axs.grid()
    axs.axis('tight')
    axs.set_title(r'Phase response')

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plot_comb_filter_tap(w,h,normalized=False,filename=None):
    fig, axs = plt.subplots(1,1,figsize=(8,8),dpi=100)
    if normalized:
        axs.plot(w/np.max(w),h_dB,'k')
        axs.set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    else:
        axs.plot(w,h_dB,'k')
        axs.set_xlabel(r'Frequency, hz')
        
    axs.set_ylabel('Magnitude (db)')
    axs.grid()
    axs.set_title(r'Frequency response, zoomed in for single tap')
    axs.axhline(y=-3.0, color="black", linestyle="--")
    axs.set_ylim((-4,0.01))
    axs.set_xlim((45,55))

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plot_impulse_response(y,filename=None):
    # plot the impulse response of a filter

    fig, axs = plt.subplots(1,1,figsize=(8,8),dpi=50)

    axs.stem(y,linefmt='-k',markerfmt='ko')

    axs.set_xlabel("n")
    axs.set_ylabel("h[n]")
    axs.set_title("Impulse Response")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

def plot_frequency(f,s,fd,ax):
    # plotting a signal's freq domain

    ax.set_xlabel('f, Hz')
    ax.set_ylabel('Sa, dB')
    #ax.set_xlim([0,1000])
    ax.set_xlim([0,fd/2])
    ax.set_ylim([-125,0])
    ax.set_title("frequency domain")
    ax.plot(f,s,'k')

def plot_amplitude(x,fd,start_delay,ax):
    # plotting a signal's freq domain
    ax.plot(x[start_delay:],'k') 
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("n")
    ax.set_title("time domain")

def plot_signal(x,fd,start_delay,title,filename=None):
    # plotting a signal in time and freq domains
    f, s = convert_to_frequency_domain(x,fd)

    fig, axs = plt.subplots(1,2,figsize=(18,8),dpi=100)
    
    plot_amplitude(x,fd,start_delay,axs[0])
    plot_frequency(f,s,fd,axs[1])
    fig.suptitle(title)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


#filters.py


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
    x = shift(x,-delay_accumulation*5, cval=0.0)

    x *= M
    # combine
    y = x_buff- x
    #y = x
    #y = x_buff - x
    #y = y + abs(y.m)
    

    y,mean,multi = normalize(y,mean=mean,multi=multi,reverse=True)

    #y = x
    # boost amplitude

    return y,x,taps




#utils.py


import numpy as np
from numpy import cos, sin, pi, absolute, arange
import scipy.signal as signal
import scipy.io

def get_inputs(n):
    # utility function for fetching the EKG signal arrays and converting to numpy

    inputs = []
    for i in range(n):
        inputs.append(scipy.io.loadmat(f'data/EKG_{i+1}.mat')["ekg"][0])
    return inputs

def get_coefs(filename):
    # utility function for fetching RIR lowpass filter coefs from a file

    coefs = []
    for i in open(filename,'r').readlines():
        coefs.append(float(i))
    return coefs

def triangle(fd,length,power,time):
    t = np.linspace(0, time, fd*time, endpoint=False)

    t,y = rectangle(fd,length,1,time)
    #y = (y+1)*0.5
    t2 = np.linspace(0, time, fd*time, endpoint=False)
    tri = signal.sawtooth(1 * np.pi * length*2 * t2,0.5)
    tri = (tri+1)*0.5
    y = tri*y
    y *= power

    return (t, y)

def rectangle(fd,length,power,time):
    t = np.linspace(0, time, fd*time, endpoint=False)
    y = signal.square(1 * np.pi * length * t)
    y = (y+1) *0.5
    y *= power

    return (t,y)

def normalize(x,mean=1,multi=1,reverse=False):
    if reverse:
        return x*multi+mean,mean,multi
    else:
        mean = x.mean()
        x = x-mean
        multi = np.max(abs(x))
        return x / multi,mean,multi

def highrate_fir_filter_parameter_calculations():
    fd = 500
    f_sl = 0.70
    f_pr = 0.51
    F = (f_sl - f_pr) / f_sl
    M = symbols('M')
    eq = Eq((f_sl**2 - f_pr**2)*M**3 - (f_sl + f_pr)
    **2*M**2 + 2*fd*(f_sl+f_pr)*M-fd**2)
    M = int(solve(eq)[0])
    
    D_1 = int((2*M*( 1 - np.sqrt( M*F / (2-F) ) )) / ( 2 - F*(M+1) ))
    D_2 = int(M/D_1)
    print(f"fd: {fd}f_sl: {f_sl}f_pr: {f_pr}")
    print(f"M: {M}, D1: {D_1}, D2: {D_2}")

def convert_to_frequency_domain(signal,fd):
    # return a frequency domain array of the signal

    nfft = len(signal)
    S = np.abs(np.fft.fft(signal) / nfft)
    S = 20 * np.log10(S/np.max(S))
    k = np.linspace(0,nfft,nfft)
    f = k*fd/nfft
    return f,S


def pad_zeros(a,f):
    b = np.zeros(len(a)*f)
    t = 0
    for i,el in enumerate(b):
        if i % f == 0:
            b[i] = a[t]
            t+=1
    return b


#main.py



# %%
from IPython import get_ipython


# %%
from utils import *
from filters import *
from analysis import *
from plots import *

# %%
def fir_test(x,fd,analyze=False):

    # fir filter
    fir_coefs = get_coefs("koefs.txt")
    y,a,b = fir_lowpass_filter(x,fir_coefs)
    if analyze: analyze_filter(b,a,fd)

    return y

def iir_test(x,fd,analyze=False,par=[0]):

    # fir comb filter
    y,a,b = comb_filter(x,fd,par) 
    if analyze: analyze_filter(b,a,fd)

    return y

def multirate_fir_test(x,fd,analyze=False,par=[0]):

    # multirate fir filter
    y,temp,tap_array = multirate_fir(x,fd,par)
    for el in analyze:
        if analyze: analyze_filter(tap_array[el],[1],fd,[2])

    return y,temp

# %%
def main():
    # initialization
    fd = 500    # sampling frequency
    rectangle_signal = rectangle(fd,10,0.003,4)
    triangle_signal = triangle(fd,5,0.0015,4)
    inputs = get_inputs(10)
    ekg_signal = inputs[4]
    par = [0]

    signal_name="EKG signal"

    comb_parameters = {}
    comb_parameters["S"] =          50          # slopinimas ties kartotine
    comb_parameters["delta_f"] =    0.67        # slopinimo plotis ties -3dB
    comb_parameters["f0"] =         50          # dazniu kartotine slopinimui
    comb_parameters["k0"] =         1           # slopinimas ties pralaidumo juosta (1=jokio)
    comb_parameters["L"] =          3           # filtro pjuvio daznis

    multirate_parameters = {}
    multirate_parameters["f_cutoff"] =              0.67    # pjuvio daznis, Hz
    multirate_parameters["f_transition_width"] =    2.5    # perejimo juostos plotis, Hz
    multirate_parameters["f_attenuation"] =         50.0     # slopinimas, dB (keicia zenkla)
    multirate_parameters["multi_attenuation"] =     50.0    # slopinimas, dB (keicia zenkla)
    #fir
    
    y_ekg_fir = fir_test(x=ekg_signal,fd=fd,analyze=[1,2,3,4,5])
    plot_signal(ekg_signal,fd,fd*2,f"{signal_name} before filtering")
    plot_signal(y_ekg_fir,fd,fd*2, f"{signal_name} after FIR filtering")

    #iir
    y_ekg_iir = iir_test(x=y_ekg_fir,fd=fd,analyze=[1,2,3,4,5],par=comb_parameters)
    plot_signal(y_ekg_iir,fd,fd*2, f"{signal_name} after comb IIR filtering")

    y_rect_iir = iir_test(x=rectangle_signal[1],fd=fd,analyze=[],par=comb_parameters)
    y_tri_iir = iir_test(x=triangle_signal[1],fd=fd,analyze=[],par=comb_parameters)
    plot_test_signal_comparison(rectangle_signal,y_rect_iir,"rectangle comb iir test",filename=None)
    plot_test_signal_comparison(triangle_signal,y_tri_iir,"triangle comb iir test",filename=None)
    
    #multirate
    #highrate_fir_filter_parameter_calculations()

    y_ekg_multirate_iir,temp = multirate_fir_test(x=ekg_signal,fd=fd,analyze=[2],par=multirate_parameters)
    plot_signal(ekg_signal,fd,fd*2, f"{signal_name} before multirate filtering")
    plot_signal(temp,fd,fd*2, f"{signal_name} drift component")
    #plt.plot(y_ekg_multirate_iir)
    #plt.plot(temp,"--r")
    #plt.show()
    plot_signal(y_ekg_multirate_iir,fd,0, f"{signal_name} after multirate filtering")

    y_rect_mr_iir,temp = multirate_fir_test(x=rectangle_signal[1],fd=fd,analyze=[],par=multirate_parameters)
    plot_signal(y_rect_mr_iir,fd,0, f"{signal_name} after multirate filtering")

    y_tri_mr_iir,temp = multirate_fir_test(x=triangle_signal[1],fd=fd,analyze=[],par=multirate_parameters)
    plot_test_signal_comparison(rectangle_signal,y_rect_mr_iir,"rectangle multirate fir test",filename=None)
    plot_test_signal_comparison(triangle_signal,y_tri_mr_iir,"triangle multirate fir test",filename=None)

def comb_testing():
    # initialization

    for i in [10]:

        comb_parameters = {}

        #comb_parameters["S"]  = 20 * np.log10(abs(1000))             # slopinimas ties kartotine
        comb_parameters["S"]  = 50             # slopinimas ties kartotine
        comb_parameters["delta_f"] = i  # slopinimo plotis ties -3dB
        
    
        # no changes
        comb_parameters["L"] = 3           # filtro pjuvio daznis
        comb_parameters["k0"] = 1          # slopinimas ties pralaidumo juosta (1=jokio)
        comb_parameters["f0"] = 50         # dazniu kartotine slopinimui
        p = comb_parameters

        fd = 500    # sampling frequency
        S = p["S"]                # slopinimas ties kartotine
        delta_f = p["delta_f"]    # slopinimo plotis ties -3dB
        f0 = p["f0"]              # dazniu kartotine slopinimui
        k0 = p["k0"]              # slopinimas ties pralaidumo juosta (1=jokio)
        L = p["L"]                # filtro pjuvio daznis
        
        # ---------------------------------------------------------------  
        
        n = fd/f0 
        k = k0*(10**(-S/20))    # slopinimas ties kf0 
        kr = k0*(10**(-L/20))   # delta_f vertinimo amplitude (-3dB pagal L)
        
        print(delta_f)
        beta = np.sqrt((kr**2-k0**2)/(k**2-kr**2)) * np.tan((n*pi*delta_f)/(2*fd))

        k1 = (k0+k*beta) / (1+beta)
        k2 = (k0-k*beta) / (1+beta)
        k3 = (1-beta) / (1+beta)

        print(beta,k1,k2,k3)

        b = np.concatenate(([k1] , np.zeros(int(n)-1) , [-k2]),axis=0)
        a = np.concatenate(([1] , np.zeros(int(n)-1) , [-k3]),axis=0)

        w,h = freqz(b=b,a=a,fs=fd)
        plot_filter_amplitude_frequency(w=w,h=h,normalized=False,filename=None)



if __name__ == "__main__":
    #comb_testing()
    main()


# %%

# %%

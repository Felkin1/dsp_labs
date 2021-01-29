import numpy as np
from numpy import cos, sin, pi, absolute, arange
import scipy.signal as signal
import scipy.io
import matplotlib.pyplot as plt
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams

def get_inputs():
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

def plot_frequency(f,s,fd,ax,sub_title=""):
    # plotting a signal's freq domain

    ax.set_xlabel('f, Hz')
    ax.set_ylabel('Sa, dB')
    #ax.set_xlim([0,1000])
    ax.set_xlim([0,fd/2])
    ax.set_ylim([-125,0])
    ax.set_title(f"{sub_title} frequency domain")
    ax.plot(f,s,'k')

def plot_amplitude(x,fd,start_delay,ax,sub_title=""):
    # plotting a signal's freq domain
    ax.plot(x[start_delay:],'k') 
    ax.set_ylabel("Amplitude")
    ax.set_xlabel("n")
    ax.set_title(f"{sub_title} time domain")

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

def plot_multiple_signals(x,fd,start_delay,title,count,titles,domains=["time","freq"],filename=None):
    # plotting a signal in time and freq domains
    fig, axs = plt.subplots(count,len(domains),figsize=(4*count,8),dpi=100,constrained_layout=True)
    for i in range(count):
        f, s = convert_to_frequency_domain(x[i],fd)
        if "time" in domains:
            if len(domains) > 1:
                plot_amplitude(x[i],fd,start_delay,axs[i][0],sub_title=titles[i])
            else:
                plot_amplitude(x[i],fd,start_delay,axs[i],sub_title=titles[i])
        if "freq" in domains:
            if len(domains) > 1:
                plot_frequency(f,s,fd,axs[i][1],sub_title=titles[i])
            else:
                plot_frequency(f,s,fd,axs[i],sub_title=titles[i])
    fig.suptitle(title)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
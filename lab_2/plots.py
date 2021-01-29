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


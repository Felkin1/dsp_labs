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
    comb_parameters["S"] =          10          # slopinimas ties kartotine
    comb_parameters["delta_f"] =    0.54        # slopinimo plotis ties -3dB
    comb_parameters["f0"] =         50          # dazniu kartotine slopinimui
    comb_parameters["k0"] =         1           # slopinimas ties pralaidumo juosta (1=jokio)
    comb_parameters["L"] =          3           # filtro pjuvio daznis

    multirate_parameters = {}
    multirate_parameters["f_cutoff"] =              0.67    # pjuvio daznis, Hz
    multirate_parameters["f_transition_width"] =    2.5    # perejimo juostos plotis, Hz
    multirate_parameters["f_attenuation"] =         50.0     # slopinimas, dB (keicia zenkla)
    multirate_parameters["multi_attenuation"] =     50.0    # slopinimas, dB (keicia zenkla)
    #fir
    first_plots = []
    titles = []

    multirate_plots = []
    multirate_titles = []

    
    y_ekg_fir = fir_test(x=ekg_signal,fd=fd,analyze=[])
    first_plots.append(ekg_signal)
    titles.append(f"{signal_name} before filtering")
    
    plot_signal(ekg_signal,fd,fd*2,f"{signal_name} before filtering")
    #plot_signal(y_ekg_fir,fd,fd*2, f"{signal_name} after FIR filtering")
    first_plots.append(y_ekg_fir)
    titles.append(f"{signal_name} after lowpass FIR filtering")
    #iir
    y_ekg_iir = iir_test(x=y_ekg_fir,fd=fd,analyze=[1,2,3,4,100],par=comb_parameters)
    plot_signal(y_ekg_iir,fd,fd*2, f"{signal_name} after comb IIR filtering")
    first_plots.append(y_ekg_iir)
    titles.append(f"{signal_name} after comb IIR filtering")
    title = "EKG filtering with a lowpass FIR filter and a comb IIR filter"
    plot_multiple_signals(first_plots,fd,fd*2,title,len(first_plots),titles,domains=["time","freq"],filename=None)
    
    y_rect_iir = iir_test(x=rectangle_signal[1],fd=fd,analyze=[],par=comb_parameters)
    y_tri_iir = iir_test(x=triangle_signal[1],fd=fd,analyze=[],par=comb_parameters)


    plot_test_signal_comparison(rectangle_signal,y_rect_iir,"rectangle comb iir test",filename=None)
    plot_test_signal_comparison(triangle_signal,y_tri_iir,"triangle comb iir test",filename=None)
    """
    #multirate
    #highrate_fir_filter_parameter_calculations()
    
    y_ekg_multirate_iir,temp0 = multirate_fir_test(x=ekg_signal,fd=fd,analyze=[0,1,2,3,4],par=multirate_parameters)
    #plot_signal(ekg_signal,fd,fd*2, f"{signal_name} before multirate filtering")
    #plot_signal(temp,fd,fd*2, f"{signal_name} drift component")
    plt.plot(y_ekg_multirate_iir)
    plt.plot(temp0,"--r")
    plt.show()
    plot_signal(y_ekg_multirate_iir,fd,0, f"{signal_name} after multirate filtering")

    y_rect_mr_iir,temp = multirate_fir_test(x=rectangle_signal[1],fd=fd,analyze=[],par=multirate_parameters)
    plot_signal(y_rect_mr_iir,fd,0, f"{signal_name} after multirate filtering")

    y_tri_mr_iir,temp = multirate_fir_test(x=triangle_signal[1],fd=fd,analyze=[],par=multirate_parameters)
    plot_test_signal_comparison(rectangle_signal,y_rect_mr_iir,"rectangle multirate fir test",filename=None)
    plot_test_signal_comparison(triangle_signal,y_tri_mr_iir,"triangle multirate fir test",filename=None)
    
    multirate_plots.append(ekg_signal)
    multirate_plots.append(temp0)
    multirate_plots.append(y_ekg_multirate_iir)

    multirate_titles.append(f"{signal_name} before multirate filtering")
    multirate_titles.append(f"{signal_name} multirate filter extracted component")
    multirate_titles.append(f"{signal_name} after multirate filtering")
    title = "EKG filtering with a multirate FIR filter"
    plot_multiple_signals(multirate_plots,fd,fd*2,title,len(multirate_plots),
    multirate_titles,domains=["time"],filename=None)
    """
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

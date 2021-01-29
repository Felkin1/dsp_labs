# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
from numpy import cos, sin, pi, absolute, arange, sqrt
import scipy.signal as signal
import scipy.io
from scipy.io.wavfile import write
from scipy.ndimage.interpolation import shift
from utils import *


# %%
class Signal:
    def __init__(self,name,data,fd):
        self.name = name
        self.data = data
        self.fd = fd
        self.n = len(data)

    def plot(self,n=None,time=True,frequency=True):
        if n is None:
            n = self.n
        plot_signal(self.data[:n],self.fd,0,f"{self.name}")


# %%
inputs = {}
fd = 8000
datapack = scipy.io.loadmat('lab3_signalai.mat')
signal_names = ['variklioSig','kabinosSig','pilotoSig']
for s in signal_names:
    inputs[s] = Signal(s,datapack[s][0],fd)


# %%
inputs

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


# %%
x = [inputs['variklioSig'].data,inputs['kabinosSig'].data,inputs['pilotoSig'].data]
fd = 8000
title = "Airplane signals for the adaptive filtering task"
titles = ["Engine Noise","Cabin signal", "Pilot Sound"]
count = len(x)
plot_multiple_signals(x,fd,0,title,count,titles,domains=["time","freq"],filename='f1.png')


# %%
inputs['variklioSig'].plot()


# %%
inputs['pilotoSig'].plot()


# %%
inputs['kabinosSig'].plot()


# %%
def mse(x,y):
    return ((x - y)**2).mean(axis=0)

class Filter:
    def __init__(self,name):
        self.name = name

class AdaptiveFilter(Filter):
    def __init__(self,name,M,miu):
        super().__init__(name)
        self.M = M
        self.miu = miu
        self.w = np.zeros(M).transpose()
        self.xa = np.zeros(M).transpose()

    def adapt_step(self,x,d,normalized=False):
        self.xa = shift(self.xa,1)
        self.xa[0] = x

        xiv = self.w.dot(self.xa)
        siv = d - xiv        
        
        if normalized:
            self.w = self.w + (self.miu / (self.xa.transpose().dot(self.xa))) * siv*self.xa
        else:
            self.w = self.w + 2*self.miu * siv*self.xa

        return siv


    def filter_signal(self,x,d,normalized=False,rmk=False):
        y = np.zeros(len(x))
        for i in range(len(x)):
            y[i] = self.adapt_step(x[i],d[i],normalized)
        return y


def mse_list(x,y,step):
    mses = []
    for i in range(len(x)):
        if i % step == 0:
            mses.append(mse(x[i-step:i],y[i-step:i]))
    return mses


# %%
dn = inputs['kabinosSig'].data # dn
xa = inputs['variklioSig'].data # xa
sn = inputs['pilotoSig'].data # sn
mius = [0.001,0.01,0.1]
Ms = [20]
mses = []
step = 800
for miu in mius:
    m0 = []
    for M in Ms:
        af = AdaptiveFilter("adaptive filter",M,miu)
        y = af.filter_signal(xa,dn,True)
        m0.append((miu,M,mse_list(sn,y,step)))
        #m0.append((miu,M,mse(sn,y)))
        print(miu,M," done")
    mses.append(m0)


# %%
mses[0][0][2]


# %%
k = ['ko','k-','k--']
i = 0
for m0 in mses:
    plt.plot(m0[0][2][:100],k[i],label=f'miu = {m0[0][0]}')
    i+=1
plt.title("nLMS algorithm applied for signal filtering at M=20 and varying $\mu$")
plt.xlabel("t, 100ms / timestep")
plt.ylabel("MSE")
plt.legend()


# %%
#x0 = [x[2] for x in mses[0]]
k = ['k','ko','k--']
i = 0
for m0 in mses:
    plt.plot([x[1] for x in m0],[x[2] for x in m0],k[i],label=f'$\mu$ = {m0[2][0]}')
    i+=1
plt.title("LMS algorithm applied for signal filtering with varying M and $\mu$ values")
plt.xlabel("M")
plt.ylabel("MSE")
plt.legend()


# %%
plt.plot(y,'r')
plt.plot(sn,'--b')
plt.show()
print(f"mse: {mse(sn,y)}")


# %%
from scipy.io.wavfile import write
write('y.wav',8000, y)
write('sn.wav',8000, sn)
write('dn.wav',8000, dn)
write('xa.wav',8000, xa)


# %%
dn = inputs['kabinosSig'].data # dn
xa = inputs['variklioSig'].data # xa
sn = inputs['pilotoSig'].data # sn
miu = 0.01
M = 20
mses = []
step = 800
af = AdaptiveFilter("adaptive filter",M,miu)
y = af.filter_signal(xa,dn)


# %%
mses = mse_list(sn,y,step)
plt.plot(mses)


# %%
dn = inputs['kabinosSig'].data # dn
xa = inputs['variklioSig'].data # xa
sn = inputs['pilotoSig'].data # sn

af = AdaptiveFilter("adaptive filter",20,0.01)
y = af.filter_signal(xa,dn)
write('y_optimal.wav',8000, y)


# %%
#dn = inputs['kabinosSig'].data # dn
#xa = inputs['variklioSig'].data # xa
#sn = inputs['pilotoSig'].data # sn

af = AdaptiveFilter("adaptive filter",20,0.01)
y = af.filter_signal(xa,dn,True)
write('y_normalized.wav',8000, y)


# %%
af.xa.transpose().dot(af.xa)


# %%
plt.plot(y,'k',label="filter output")
plt.plot(sn,'k',label='ground truth',c='0.75')
plt.title("LMS algorithm's output comparison to ground truth")
plt.ylabel("amplitude")
plt.xlabel("t, 8000Hz sampling rate")
plt.legend()
plt.show()

print(f"mse: {mse(sn,y)}")


# %%
class AdaptiveRMKFilter(Filter):
    def __init__(self,name,M,miu):
        super().__init__(name)
        self.M = M
        self.miu = miu
        self.w = np.zeros(M).transpose()
        self.xa = np.zeros(M).transpose()
        self.gamma = 0.01
        self.I = np.identity(M)
        self.P = self.I / self.gamma

        self.lambda0 = 1

    def norm(self,x):
        return np.matmul(x.transpose(),x)


    def adapt_step(self,x,d):
        self.xa = shift(self.xa,1)
        self.xa[0] = x
        xa = self.xa.reshape(self.M,1)
        P = self.P
        w = self.w.reshape(self.M,1)
        l = self.lambda0


        #print(f'l: {l}\nxa:\n{xa} \nP:\n{P}\nw:\n{w}')

        # RMS algorithm

        # np.matmul     returns matrix product of two given arrays
        # np.multiply   returns element-wise multiplication of two given arrays
        # np.dot        returns scalar or dot product of two given arrays

        v = np.matmul(P,xa)       
        u = np.matmul(P.transpose(),v)        
        #print(v)
        #print(v.shape)
        v = v.reshape(len(v),1)
        #print(v.shape)
        k = 1 / ( l + self.norm(v) + sqrt(l) * sqrt( l + self.norm(v) ) )
        
        xiv = np.matmul(w.transpose(),xa)
        siv = d - xiv    

        self.P = ( P - k * np.matmul(v,u.transpose())) / sqrt(l)    
        self.w = w + (siv*u)/(l + self.norm(v))
        #print(w.shape,xa.shape,v.shape,u.shape)
        #print("\nafter calculations\n")
        #print(f"\nv: {v} \nu: {u} \nk:{k} \nnew P: \n{self.P} \nxiv:\n{xiv}\nnew w:\n{self.w}")
        return siv

    def filter_signal(self,x,d):
        y = np.zeros(len(x))
        for i in range(len(x)):
            #print(f"\niteration: {i}")
            y[i] = self.adapt_step(x[i],d[i])
        return y

dn = inputs['kabinosSig'].data # dn
xa = inputs['variklioSig'].data # xa
sn = inputs['pilotoSig'].data # sn



af = AdaptiveRMKFilter("adaptive filter",20,0.01)
#y = af.filter_signal(xa[0:20],dn[0:20])
y = af.filter_signal(xa,dn)
#y = af.filter_signal(xa[:2],dn[:2])
#write('y_rmk.wav',8000, y)


# %%
plt.plot(y[100:],'r')
plt.plot(sn[100:],'b--')
plt.show()

print(f"mse: {mse(sn,y)}")


# %%
write('y_rmk.wav',8000, y)


# %%
dn = inputs['kabinosSig'].data # dn
xa = inputs['variklioSig'].data # xa
sn = inputs['pilotoSig'].data # sn
mius = [0.95,0.96,0.97,0.98,0.99,1]
Ms = [2,4,6,8,10,15,20,30,40,60]
mses = []
step = 800
for miu in mius:
    m0 = []
    for M in Ms:
        #af = AdaptiveFilter("adaptive filter",M,miu)
        af = AdaptiveRMKFilter("adaptive filter",M,miu)
        y = af.filter_signal(xa,dn)
        #m0.append((miu,M,mse_list(sn,y,step)))
        m0.append((miu,M,mse(sn,y)))
        print(miu,M," done")
    mses.append(m0)


# %%
k = ['k','ko','k--','r','ro','r--']
i = 0
for m0 in mses:
    plt.plot([x[1] for x in m0],[x[2] for x in m0],k[i],label=f'$\lambda$ = {m0[2][0]}')
    i+=1
plt.title("RMS algorithm applied for signal filtering with varying M and $\lambda$ values, $\gamma$=0.01")
plt.xlabel("M")
plt.ylabel("MSE")
plt.legend()


# %%
dn = inputs['kabinosSig'].data # dn
xa = inputs['variklioSig'].data # xa
sn = inputs['pilotoSig'].data # sn
mius = [0.01]
Ms = [20]
mses = []
step = 800
for miu in mius:
    m0 = []
    for M in Ms:
        #af = AdaptiveFilter("adaptive filter",M,miu)
        af = AdaptiveRMKFilter("adaptive filter",M,miu)
        y = af.filter_signal(xa,dn)
        #m0.append((miu,M,mse_list(sn,y,step)))
        m0.append((miu,M,mse(sn,y)))
        print(miu,M," done")
    mses.append(m0)


# %%
k = ['k','k3','k--','k.','k^']
i = 0
for m0 in mses:
    plt.plot(m0[0][2][:100],k[i],label=f'miu = {m0[0][0]}')
    i+=1
plt.title("RMS algorithm applied for signal filtering at M=20, $\lambda$=1 and varying $\gamma$")
plt.xlabel("t, 100ms / timestep")
plt.ylabel("MSE")
plt.yscale('log')
plt.legend()


# %%




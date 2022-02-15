"""Process some integers.

usage: crying_vegetable_jikken [-h] (--tsv=<jikken_tsv>) (--prefix=<time_prefix>) [--type=<type_out>]

options:
    -h, --help  show this help message and exit
    --tsv=<jikken_tsv>        data file name
    --prefix=<time_prefix>    time line prefix string
    --type=<type_out>         wave output type, 'chirp', 'male_a', ..., 'male_o', 'female_a', ..., 'female_o' [default: 'chirp'] 
"""

import fire
# %% [markdown]
# # 時系列計測データを音声に

# %%
import sys
import soundfile as sf
import numpy as np
from scipy import signal as scisig
from scipy import fftpack as scifft
from kslib import vocoder, reduct_frac
from kslib.mattoolbox import signal as matsig
from . import vegetablecry as vc
from IPython.display import Audio
from matplotlib import pyplot as plt
import japanize_matplotlib
#import SciencePlots
plt.style.use(['science', 'ieee'])
#plt.style.use(['science', 'cjk-jp-font', 'no-latex'])
from matplotlib import rc
#rc('text', usetex=True)

from os import path

def jikken(jikken_tsv,time_prefix,type_out,fs_out=44100,fs_data=24,time_unit='minutes'):


    # %%
    #jikken_tsv = 'data/jikken2-a.txt'
    #time_prefix = 'jikken8.200'
    #jikken_tsv = 'data/jikken3.txt'
    #time_prefix = 'jikken300'

    #time_unit = 'minutes'


    # %%

    # duration_change_rate
    #fs_data = 24       # in [1/s] 

    # %%
    # 出力する音響信号について
    #fs_out = 44100      # in [1/s]

    #type_out = 'chirp'
    #type_out = 'male_a'
    #type_out = 'male_i'
    #type_out = 'male_u'
    #type_out = 'male_e'
    #type_out = 'male_o'
    #type_out = 'female_a'
    #type_out = 'female_i'
    #type_out = 'female_u'
    #type_out = 'female_e'
    #type_out = 'female_o'



    # %%

    jikken_prefix = path.splitext(path.basename(jikken_tsv))[0]

    t, f = vc.read_jikken(jikken_tsv,time_prefix,time_unit) 
    tr, fr = vc.resample_jikken(t,f) # 欠損補間

    fs_data0 = 1/(tr[1]-tr[0])


    # %%
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(tr,fr)
    ax1.set_ylim([-1,1])
    ax1.set_xlabel('Time [sec.]')
    ax1.set_ylabel('NDI [?]')
    ax1.set_title(jikken_tsv)
    ax1.grid()
    TimeVariantNDI_png = 'result/fig/' + jikken_prefix + '_fig_TimeVariantNDI.png'
    fig.savefig(TimeVariantNDI_png)

    # %%
    if type_out == 'chirp':
        gender = 'chirp'
        vowel = 'broad'
        target_freq_limits = [200,3500]
    else:
        gender, vowel = type_out.split('_')
        if gender == 'female':
            target_freq_limits = [168, 880]
        elif gender == 'male':
            target_freq_limits = [84, 440]

    jikken_wav = 'result/wav/' + jikken_prefix + '_' + gender + '_' + vowel + '.wav'

    print(gender,vowel,target_freq_limits)

    # %%
    f2 = matsig.scaling(fr, [-1,+1], target_freq_limits)

    # %%
    (p,q) = reduct_frac.reduct_frac(fs_out,fs_data)
    f2ud = matsig.upsampling(f2,p,q)

    t2 = np.arange(len(f2ud))/fs_out

    # %%
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t2,f2ud)
    ax.grid()
    ax.set_ylim(target_freq_limits)
    ax.set_xlabel('Time [sec.]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(f'Time variant frequencies of Vocoded Vegetable Cry\n{gender}-/{vowel}/')
    TimeVariantFreq_png = 'result/fig/' + jikken_prefix + f'_fig_TimeVariantFreq_{gender}_{vowel}.png'
    fig.savefig(TimeVariantFreq_png)

    # %%
    instant_phase = np.cumsum(f2ud)/fs_out
    y = np.sin(2*np.pi*instant_phase)

    if type_out != 'chirp':
        vocal = np.full((len(t2),),0.0)
        vocal[np.where(y>0.95)[0]]=1.0
        y = vocoder.formant_filter(vocal,fs_out,gender,vowel)

    # %%
    time_region = [2.5, 2.6]
    point_region = np.array(time_region)*fs_out
    point_region = point_region.astype('int')
    print(f'{point_region}')
    V = scifft.fft(y[point_region[0]:point_region[1]])
    lx = len(V)
    Mag = 10*np.log10(np.real(V*np.conj(V)))
    freq = np.arange(lx)/lx*fs_out
    nlx = lx//2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.loglog(freq[:nlx],Mag[:nlx])
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_ylim([10**0, 10**2])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Power [dB]')
    ax.set_title(f'Power Spectrum {time_region[0]}--{time_region[1]}[sec.]')
    ax.grid()

    # %%
    y /= np.amax(np.abs(y))
    y *= 0.8 
    sf.write(jikken_wav,y,fs_out)
    Audio(jikken_wav, rate=fs_out)

def main():
    fire.Fire(jikken)
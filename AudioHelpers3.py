import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import os
import multiprocessing
import datetime

path = (os.getcwd() + '\\training\\')

def blit(dest, src, loc):
    pos = [i if i >= 0 else None for i in loc]
    neg = [-i if i < 0 else None for i in loc]
    target = dest[[slice(i,None) for i in pos]]
    src = src[[slice(i, j) for i,j in zip(neg, target.shape)]]
    target[[slice(None, i) for i in src.shape]] = src
    return dest

def convert_stereo(lock, filename, return_dict):

    rate, fileData = wavfile.read(path + filename)

    data = np.memmap(filename + ".tmp", mode='w+', shape=(2097152, 6), dtype=np.int32)

    blit(data, fileData, (0,0))

    leftChannels = np.delete(data, np.s_[1,3,5], axis=1)
    rightChannels = np.delete(data, np.s_[0,3,4], axis=1)

    leftCombined = np.sum(leftChannels, axis=1)/3 
    rightCombined = np.sum(rightChannels, axis=1)/3

    newData = np.stack((leftCombined, rightCombined), axis=-1)

    amplification = 1 / np.max(newData)

    data = data #* amplification
    newData = newData# * amplification

    spectogram(rate, data[:,0])

    return_dict[filename + 'old'] = data
    return_dict[filename + 'new'] = newData
    
    wavfile.write(path + "stereo\\" + filename, rate, newData)

def spectogram(rate, data):
    cmap = plt.get_cmap('viridis') # this may fail on older versions of matplotlib
    vmin = -40  # hide anything below -40 dB
    cmap.set_under(color='k', alpha=None)
    fig, ax = plt.subplots()
    pxx, freq, t, cax = ax.specgram(data[:, 0], # first channel
                                    Fs=rate,      # to get frequency axis in Hz
                                    cmap=cmap, vmin=vmin)
    cbar = fig.colorbar(cax)
    cbar.set_label('Intensity dB')
    ax.axis("tight")

    ax.set_xlabel('time h:mm:ss')
    ax.set_ylabel('frequency kHz')

    scale = 1e3                     # KHz
    ticks = matplotlib.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale))
    ax.yaxis.set_major_formatter(ticks)

    def timeTicks(x, pos):
        d = datetime.timedelta(seconds=x)
        return str(d)
    formatter = matplotlib.ticker.FuncFormatter(timeTicks)
    ax.xaxis.set_major_formatter(formatter)
    plt.show()
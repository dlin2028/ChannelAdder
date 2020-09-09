import numpy as np
import matplotlib.pyplot as plot
from scipy.io import wavfile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import os
import multiprocessing
import datetime



def prepare_training(path, filenames):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    lock = multiprocessing.Lock()
    procs = []

    for filename in filenames:
        if not filename.endswith(".wav"):
            continue

        print("found song " + filename)

        lock.acquire()
        p = multiprocessing.Process(target=convert_stereo, args=(lock, path, filename, return_dict))
        procs.append(p)
        p.start()
        lock.release()

    for p in procs:
        p.join()

    print("done converting to stereo")
    return return_dict;

def blit(dest, src, loc):
    pos = [i if i >= 0 else None for i in loc]
    neg = [-i if i < 0 else None for i in loc]
    target = dest[[slice(i,None) for i in pos]]
    src = src[[slice(i, j) for i,j in zip(neg, target.shape)]]
    target[[slice(None, i) for i in src.shape]] = src
    return dest


def read_inputs(path, filenames):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    lock = multiprocessing.Lock()
    procs = []

    for filename in filenames:
        if not filename.endswith(".wav"):
            continue

        print("found song " + filename)

        lock.acquire()
        p = multiprocessing.Process(target=read_wav, args=(lock, path, filename, return_dict))
        procs.append(p)
        p.start()
        lock.release()

    for p in procs:
        p.join()

    print("done reading inputs")
    return return_dict;

def read_wav(lock, path, filename, return_dict):
    rate, fileData = wavfile.read(path + filename)
    data = np.memmap(filename + ".tmp", mode='w+', shape=(2097152//4, 2), dtype=np.int32)
    blit(data, fileData, (0,0))
    
    amplification = 1 / np.max(data)
    data = data * amplification

    return_dict[filename] = data
    return_dict[filename + "rate"] = rate

def write_wav(path, filename, rate, data):
    wavfile.write(path + filename, rate, data)

def convert_stereo(lock, path, filename, return_dict):

    rate, fileData = wavfile.read(path + filename)

    data = np.memmap(filename + ".tmp", mode='w+', shape=(2097152//4, 6), dtype=np.int32)

    blit(data, fileData, (0,0))

    leftChannels = np.delete(data, np.s_[1,3,5], axis=1)
    rightChannels = np.delete(data, np.s_[0,3,4], axis=1)

    leftCombined = np.sum(leftChannels, axis=1)/3 
    rightCombined = np.sum(rightChannels, axis=1)/3

    newData = np.stack((leftCombined, rightCombined), axis=-1)
  
    amplification = 1 / np.max(data)

    data = data * amplification
    newData = newData * amplification

    return_dict[filename + 'old'] = data
    return_dict[filename + 'new'] = newData

    wavfile.write(path + "\\stereo\\" + filename, rate, newData)
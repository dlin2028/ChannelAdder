import warnings
warnings.filterwarnings("ignore")

import os
import multiprocessing
from pathlib import Path

import AudioHelpers as ah
import kerasHelpers
kh = kerasHelpers.kerasHelpers()

import numpy as np

if __name__ == '__main__':
    
    samples = 500;
    text = input("train or predict\n")
    
    if text == "train":
        path = os.getcwd() + '\\training\\'
        filenames = os.listdir(path)
        sound_data = ah.prepare_training(path, filenames)
    
        for filename in filenames:
            if not filename.endswith(".wav"):
                continue
        
            stereo = np.array(sound_data[filename + "new"])
            surround = np.array(sound_data[filename + "old"])

            kh.create_model(samples)
            kh.train(stereo, surround, samples, 4)
    else:
        path = os.getcwd()
        filenames = os.listdir(path + '\\inputs\\')
        
        kh.load_model(path, '\\model.h5', samples)
        sound_data = ah.read_inputs(path + '\\inputs\\', filenames)

        for filename in filenames:
            if not filename.endswith(".wav"):
                continue

            print("upscaling " + filename)

            prediction = kh.predict(sound_data[filename], samples)
            
            rate = sound_  data[filename + "rate"]
            data = sound_data[filename]
            
            ah.write_wav(path + '\\outputs\\', filename, rate, np.array(prediction))
            print("done converting")
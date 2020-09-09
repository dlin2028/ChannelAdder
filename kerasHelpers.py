import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import numpy as np

class kerasHelpers(object):
    """description of class"""

    def create_model(self, samples):
        # This returns a tensor
        inputs = Input(shape=(samples * 2,2))

        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(128, activation='relu')(inputs)
        lstm_out = LSTM(32)(x)
        
        x = Dense(64, activation='relu')(lstm_out)
        a = Dense(1, activation='sigmoid', name='a')(x)

        
        x = Dense(64, activation='relu')(lstm_out)
        b = Dense(1, activation='sigmoid', name='b')(x)

        
        x = Dense(64, activation='relu')(lstm_out)
        c = Dense(1, activation='sigmoid', name='c')(x)

        
        x = Dense(64, activation='relu')(lstm_out)
        d = Dense(1, activation='sigmoid', name='d')(x)

        
        x = Dense(64, activation='relu')(lstm_out)
        e = Dense(1, activation='sigmoid', name='e')(x)

        
        x = Dense(64, activation='relu')(lstm_out)
        f = Dense(1, activation='sigmoid', name='f')(x)



        # This creates a model that includes
        # the Input layer and three Dense layers
        self.model = Model(inputs=inputs, outputs=[a,b,c,d,e,f])
        self.model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

    def train(self, stereo, surround, samples, batch):
        stereo_data = []
        surround_data = []

        for i in range(samples, len(stereo) - samples):
            stereo_data.append(stereo[i - samples:i + samples])

        for i in range(0, len(surround[0])):
            surround_data.append((surround[:,i])[samples:-samples])

        stereo_batches = np.array_split(stereo_data, batch)
        surround_batches = np.hsplit(np.array(surround_data), batch)

        for i in range(batch):
            ster = stereo_batches[i]
            surr = [np.array(xi) for xi in surround_batches[i]]

            self.model.fit(ster, surr, epochs=5, batch_size=500)
            self.model.evaluate(ster, surr)

            # serialize weights to HDF5
            self.model.save_weights("model.h5")
            print("Saved model to disk")
            
    def load_model(self, path, filename, samples):
        self.create_model(samples)
        self.model.load_weights(path + filename)

    def predict(self, input_data, samples):
        stereo_data = []

        for i in range(samples, len(input_data) - samples):
            stereo_data.append(input_data[i - samples:i + samples])

        prediction = self.model.predict(np.array(stereo_data), batch_size=1000, verbose=True)
        
        surround_output = np.stack(prediction, axis=1)

        return surround_output;
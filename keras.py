class kerasHelpers(object):
    """description of class"""
    
    def create_model(self):
        # This returns a tensor
        inputs = Input(shape=(samples * 2,2))

        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        lstm_out = LSTM(32)(x)
        
        x = Dense(128, activation='relu')(lstm_out)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        a = Dense(1, activation='sigmoid', name='a')(x)

        
        x = Dense(128, activation='relu')(lstm_out)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        b = Dense(1, activation='sigmoid', name='b')(lstm_out)

        
        x = Dense(128, activation='relu')(lstm_out)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        c = Dense(1, activation='sigmoid', name='c')(lstm_out)

        
        x = Dense(128, activation='relu')(lstm_out)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        d = Dense(1, activation='sigmoid', name='d')(lstm_out)

        
        x = Dense(128, activation='relu')(lstm_out)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        e = Dense(1, activation='sigmoid', name='e')(lstm_out)

        
        x = Dense(128, activation='relu')(lstm_out)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        f = Dense(1, activation='sigmoid', name='f')(lstm_out)



        # This creates a model that includes
        # the Input layer and three Dense layers
        self.model = Model(inputs=inputs, outputs=[a,b,c,d,e,f])
        sel.fmodel.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['accuracy'])

    def train(self, stereo, surround, samples, batch):

        samples = 50
        batch = 4

        stereo_data = []
        surround_data = []

        for i in range(samples, len(stereo) - samples):
            stereo_data.append(stereo[i-samples:i+samples])

        for i in range(0, len(surround[0])):
            surround_data.append((surround[:,i])[samples:-samples])

        stereo_batches = np.array_split(stereo_data, batch)
        surround_batches = np.hsplit(np.array(surround_data), batch)

        for i in range(batch):
            ster = stereo_batches[i]
            surr = [np.array(xi) for xi in surround_batches[i]]

            model.fit(ster, surr, epochs=5)
            model.evaluate(ster, surr)

            # serialize model to JSON
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("model.h5")
            print("Saved model to disk")
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, TimeDistributed
from keras.layers import Input, Conv1D, BatchNormalization, Activation, GRU
from keras.optimizers import Adam

from librosa import feature, core
from tqdm import tqdm

sample_rate = 44100
feed_duration = 5
feed_samples = int(sample_rate * feed_duration)

def model(input_shape):
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer
    X = Conv1D(98, kernel_size=15, strides=4)(X_input)    # CONV1D
    X = BatchNormalization()(X)                           # Batch normalization
    X = Activation('relu')(X)                             # ReLu activation
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)

    # Step 2: First GRU Layer
    X = GRU(units = 64, return_sequences = True)(X)      # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)
    X = BatchNormalization()(X)                           # Batch normalization
    
    # Step 3: Second GRU Layer
    X = GRU(units = 64, return_sequences = True)(X)      # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)
    X = BatchNormalization()(X)                           # Batch normalization
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)
    model = Model(inputs = X_input, outputs = X)    
    return model

def train_test_split(x, y, ratio=0.8):
    n = int(len(x) * ratio)
    return (x[:n], y[:n], x[n:], y[n:])

if __name__ == "__main__":
  with open('./data/info.json') as f:
    info = json.load(f)

  info = list(info.items())
  random.shuffle(info)

  print('-' * 16)
  print('Setup training data')
  Tx, n_freq = 431, 40
  Ty = int((Tx - 15) / 4) + 1
  X_data, y_data = [], []
  for path, val in tqdm(info):
    x, sr = core.load(os.path.join('./data/sample', path), sr=sample_rate)
    m = feature.mfcc(x[:feed_samples], sr=sample_rate, n_mfcc=n_freq+1).T[:,1:] # shape = (431,32)
    # print(m.shape)
    X_data.append(m)
    p = int(val * Ty)
    y = np.zeros((Ty, 1))
    y[p - 10:p, 0] = 1
    y_data.append(y)
  
  X_data, y_data = np.array(X_data), np.array(y_data)

  X_train, y_train, X_test, y_test = train_test_split(X_data, y_data, 0.8)
  
  print('-' * 16)
  print('Establish model')
  model = model(input_shape = (Tx, n_freq))
  opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
  model.summary()

  print('-' * 16)
  print('Training start')
  hist = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_split=0.2)

  loss, acc = model.evaluate(X_test, y_test)
  print("Test Score: {0}\nTest Accuracy: {1}".format(loss, acc))

  # Save weights
  model.save('./data/trained.h5')

  # Loss History
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

'''
y, sr = core.load('./data/sample/0000.mp3', sr=44100)
m = feature.mfcc(y, sr, n_mfcc=13)
print(m.shape)

plt.imshow(np.flip(m[1:,], 0), cmap='jet', aspect=8.0)
plt.xlabel('time (s)')
plt.ylabel('MFCC Coefficients')
plt.show()
'''
import os
import pyaudio
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from threading import Thread

from librosa import feature, core

import keras
from keras.models import load_model
from keras import backend as K

sample_rate = 44100
chunk_duration = 0.3 # Each read length in seconds from mic.
chunk_samples = int(sample_rate * chunk_duration) # Each read length in number of samples.
# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 5
feed_samples = int(sample_rate * feed_duration)
Tx, n_freq = 431, 40

def get_spectrogram(x):
  spec = feature.mfcc(x, sr=sample_rate, n_mfcc=n_freq+1).T[:,1:]
  return spec

def get_audio_input_stream(callback):
  stream = pyaudio.PyAudio().open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=sample_rate,
    input=True,
    frames_per_buffer=chunk_samples,
    input_device_index=0,
    stream_callback=callback)
  return stream

def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.7):
  predictions = predictions > threshold
  chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
  chunk_predictions = predictions[-chunk_predictions_samples:]
  level = chunk_predictions[0]
  for pred in chunk_predictions:
    if pred > level:
      return True
    else:
      level = pred
  return False

if __name__ == "__main__":
  model = load_model('./data/trained.h5')

  def detect_triggerword_spectrum(x):
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    return pred.reshape(-1)
  
  # Queue to communiate between the audio callback and main thread
  q = Queue()
  run = True
  silence_threshold = 100/65535
  # Run the demo for a timeout seconds
  timeout = time.time() + 0.5*60  # 0.5 minutes from now
  # Data buffer for the input wavform
  data = np.zeros(feed_samples, dtype='float32')
  
  def callback(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold
    if time.time() > timeout:
        run = False        
    data0 = np.frombuffer(in_data, dtype='float32')
    if np.abs(data0).mean() < silence_threshold:
        # sys.stdout.write('-')
        return (in_data, pyaudio.paContinue)
    # else:
    #     sys.stdout.write('.')
    data = np.append(data, data0)
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)
  
  print('Listening')
  stream = get_audio_input_stream(callback)
  stream.start_stream()

  try:
    while run:
      data = q.get()
      spec = get_spectrogram(data)
      pred = detect_triggerword_spectrum(spec)
      new_trigger = has_new_triggerword(pred, chunk_duration, feed_duration)
      if new_trigger:
        print('called me')# sys.stdout.write('1')
  except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False

  stream.stop_stream()
  stream.close()
  print('Bye bye!')


import os
import json
import numpy as np
from random import randint
from pydub import AudioSegment
from tqdm import tqdm

clip_length = 5000

def get_segment(data, segment):
  sta = randint(0, len(data) - segment)
  return data[sta : sta+segment]

def add_segment(src, dst):
  sta = randint(0, len(dst) - len(src) - 25)
  return dst.overlay(src, position=sta, gain_during_overlay=-30), sta + len(src) - 1

def make_sample(positive, negative, background):
  # take random position from negative sample
  sample = get_segment(background, clip_length) - randint(0,10)
  base = get_segment(negative, clip_length)
  base, idx = add_segment(positive, base)
  return sample.overlay(base - randint(0,10)), float(idx) / clip_length

def get_filelist(basedir):
  files = []
  for path in os.listdir(basedir):
    if path.lower().endswith(('.mp3', '.m4a', '.wav')):
      files.append(os.path.join(basedir, path))
  return files

if __name__ == "__main__":
  outdir = './data/sample'

  backgrounds = get_filelist('./data/background')
  negatives = get_filelist('./data/negative')
  positives = get_filelist('./data/positive')

  info = {}
  count = 0
  for background in tqdm(backgrounds):
    bgm = AudioSegment.from_file(background)
    for negative in tqdm(negatives):
      neg = AudioSegment.from_file(negative)
      for positive in tqdm(positives):
        pos = AudioSegment.from_file(positive)
        res, val = make_sample(pos, neg, bgm)
        path = '%04d.mp3' % count
        res.export(os.path.join(outdir, path))
        info[path] = val
        count += 1

  with open('./data/info.json', 'w') as f:
    json.dump(info, f)

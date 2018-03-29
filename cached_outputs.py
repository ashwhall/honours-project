import os
import pickle

class CachedOutputs:
  OUTPUT_DIR = os.path.join('bin', 'cached_outputs')

  @staticmethod
  def output_filepath(filename):
    new_filename = filename.replace(os.sep, '_') + '.bin'
    return os.path.join(CachedOutputs.OUTPUT_DIR, new_filename)

  @staticmethod
  def save(filename, obj):
    if not os.path.isdir(CachedOutputs.OUTPUT_DIR):
      os.makedirs(CachedOutputs.OUTPUT_DIR)
    with open(CachedOutputs.output_filepath(filename), 'wb') as f:
      pickle.dump(obj, f)

  @staticmethod
  def load(filename):
    with open(CachedOutputs.output_filepath(filename), 'rb') as f:
      return pickle.load(f)

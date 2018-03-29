import threading
import numpy as np

from constants import Constants
# Keywords for indicating the dataset mode
TRAIN = 'train'
TEST = 'test'
EVAL = 'eval'
MODES = [TRAIN, TEST, EVAL]

class DataInterface:
  '''
  The interface for pulling batches of frames. Asynchronously puts batches into
  `batch_queue`
  '''
  def __init__(self, _data_loader):
    self._data_loader = _data_loader

  def num_classes(self, dataset):
    '''
    Returns the number of classes in the dataset
    '''
    return self._data_loader.num_classes(dataset)

  def get_next_batch(self, dataset, indices, num_shot, query_size=1):
    '''
    Builds a support/query batch and returns it
    '''
    return self._data_loader.get_next_batch(dataset, indices, num_shot, query_size)

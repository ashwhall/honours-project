import queue
import random
import threading
import multiprocessing
import numpy as np

from constants import Constants
from .batch import Batch
from .random_queue import RandomQueue


class DataInterface:
  '''
  The interface for pulling batches of frames. Asynchronously puts batches into
  `batch_queue`
  '''
  def __init__(self, global_sampler, queue_size):
    self.global_sampler = global_sampler
    self.batch_size = Constants.config['batch_size']

    self.batch_item_queue = RandomQueue(queue_size)
    self.batch_queue = queue.Queue(max(1, queue_size))
    self.batch_item_queue_thread = threading.Thread(target=self._fill_batch_item_queue)
    self.batch_item_queue_thread.daemon = True
    self.batch_item_queue_thread.start()
    self.batch_queue_thread = threading.Thread(target=self._fill_batch_queue)
    self.batch_queue_thread.daemon = True
    self.batch_queue_thread.start()


  def get_next_batch(self):
    '''
    Returns the next batch from the queue
    '''
    batch = self.batch_queue.get()
    return batch

  def _add_to_batch_item_queue(self, image, target):
    '''
    Creates an example and adds it to the example queue
    '''
    batch_item = (image, target)
    self.batch_item_queue.put(batch_item)

  def _fill_batch_item_queue(self):
    '''
    Fills the example queue with consecutive frames.
    '''
    while True:
      self._add_to_batch_item_queue(*self.global_sampler.next())

  def _add_to_batch_queue(self, batch_items):
    '''
    Creates a batch, does any augmentation on it and adds it to the batch queue
    '''
    batch = Batch(batch_items)
    self.batch_queue.put(batch)

  def _fill_batch_queue(self):
    '''
    Grabs elements from batch_item queue and creates batch objects
    '''
    while True:
      batch_items = []
      for _ in range(self.batch_size):
        batch_items.append(self.batch_item_queue.get())
      self._add_to_batch_queue(batch_items)

  def join(self):
    '''
    Joins all threads that this object makes
    '''
    self.batch_item_queue_thread.join()
    self.batch_queue_thread.join()

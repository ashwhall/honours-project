import csv
import functools
import os
import pickle
import queue
import random
import sys
import threading
import numpy as np
import tensorflow as tf

from base_runner import BaseRunner
from cached_outputs import CachedOutputs
from constants import Constants
from helper import Helper

import data_loader.data_partitioner as data_partitioner
from data_loader.data_interface import DataInterface

class Forward(BaseRunner):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.graph_nodes = self.graph_builder.build_graph(only_test=True)
    self._build_datasets()

  def _build_datasets(self):
    '''
    Retrieve the train/test datasets
    '''
    self.datasets = data_partitioner.load_datasets('datasets',
                                                   Constants.config['dataset'],
                                                   Constants.config['num_way'] if 'fixed_classes' in Constants.config else None)
    self.test_set  = DataInterface(self.datasets['test'], 'eval', Constants.config['num_way'] if 'fixed_classes' in Constants.config else None)

  def _test_pass(self, support_set, query_set):
    '''
    A single pass through the given batch from the test set
    Note that _test_pass and _training_pass are separate functions in case their internals diverge.
    '''
    loss, outputs = self.sess.run([
        self.graph_nodes['loss'],
        self.graph_nodes['outputs'],
    ], {
        self.graph_nodes['support_images']: support_set['images'],
        self.graph_nodes['query_images']: query_set['images'],
        self.graph_nodes['input_y']: query_set['labels'],
        self.graph_nodes['is_training']: False
    })
    return loss, outputs

  def run(self):
    test_set = self.test_set
    self._start_tf_session()

    num_test_passes = 1000
    results = {}

    for num_way in Constants.config['eval_num_way']:
      print("{}-way - {} steps".format(num_way, num_test_passes))
      predictions = []
      targets = []
      losses = []
      for i in range(num_test_passes):
        print("{:.2f}%\r".format(100 * i / num_test_passes), end='', flush=True)
        support_set, query_set = test_set.get_next_batch(num_way=num_way)
        loss, outputs = self._test_pass(support_set, query_set)
        predictions.append(outputs)
        targets.append(query_set['labels'])
        losses.append(loss)
      results[num_way] = {
          'predictions': predictions,
          'targets': targets,
          'losses': losses,
      }
      print("100.00%")

    print("Complete")
    CachedOutputs.save('model_outputs', results)

def main(argv):
  np.set_printoptions(precision=4, linewidth=150, suppress=True)

  if '--config_file' in argv:
    config_file = argv[argv.index('--config_file') + 1]
  else:
    config_file = 'basic_config.yml'

  if '--description' in argv:
    description = argv[argv.index('--description') + 1]
  else:
    description = ''

  random.seed(7218)
  tf.set_random_seed(6459)
  np.random.seed(7518)

  forward = Forward(config_file, description, argv)
  forward.run()


if __name__ == "__main__":
  main(sys.argv)

import re
import collections
import os
import csv
import random
import sys
import time
import threading
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime

import tensorflow as tf
import numpy as np

from base_runner import BaseRunner
from constants import Constants
from helper import Helper
import data_loader.data_partitioner as data_partitioner
from data_loader.data_interface import DataInterface
from data_loader.labels.label import Label

# Define some application flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('write_logs', False, 'If True, writes tensorboard logs. Default: False')

datasets = None
data_interface = None

class SourceTrainer(BaseRunner):
  '''
  This is the entry point for training a model. Takes care of loading config file,
  model, loads from file (if saved) etc.
  '''
  def __init__(self, config_file, description, class_indices, bin_dir, data_interface):
    super().__init__(config_file, description=description, argv=None, bin_dir=bin_dir)
    self.graph_nodes, self._model_module = self.graph_builder.build_graph()
    self._class_indices = class_indices
    self.data_interface = data_interface

  def run(self):
    '''
    Starts the training process - the workhorse
    '''
    start_time = time.time()
    self._start_tf_session()

    self._run_training()

    print("Training complete. \n\tTime taken: {:6.2f} sec".format(time.time() - start_time))
    final_test_score = self._get_final_test_score()
    self.sess.close()
    if FLAGS.write_logs:
      self.writer.close()
    tf.reset_default_graph()
    Helper.reset_tb_data()
    return final_test_score

  def _get_final_test_score(self):
    top_ones = []
    for _ in range(20):
      images, labels = self.data_interface.get_next_batch('test', self._class_indices, num_shot=Constants.config['num_shot'])
      top_ones.append(self.sess.run([
          self.graph_nodes['top_1']
      ], {
          self.graph_nodes['images']: images,
          self.graph_nodes['labels']: labels,
          self.graph_nodes['is_training']: False
      }))

    return np.mean(top_ones)

  def _training_pass(self):
    '''
    A single pass through the given batch from the training set
    '''
    images, labels = self.data_interface.get_next_batch('train', self._class_indices, num_shot=Constants.config['num_shot'])
    loss, outputs, summary = self._model_module.training_pass(self.sess, self.graph_nodes, self.graph_nodes['source_train_summary_op'], images, labels)
    return loss, outputs, summary

  def _test_pass(self):
    '''
    A single pass through the given batch from the training set
    '''
    images, labels = self.data_interface.get_next_batch('test', self._class_indices, num_shot=Constants.config['num_shot'])
    loss, outputs, summary = self._model_module.test_pass(self.sess, self.graph_nodes, self.graph_nodes['source_test_summary_op'], images, labels)
    return loss, outputs, summary

  def _run_training(self):
    '''
    The training/validation loop
    '''
    summary_freq = 100
    test_pass_freq = 50
    step = self.sess.run(self.graph_nodes['global_step'])
    last_pass_test = False
    while step < Constants.config['total_steps']:
      loop_start_time = time.time()

      if step % test_pass_freq == 0 and not last_pass_test:
        loss, outputs, summary = self._test_pass()
        last_pass_test = True
      else:
        loss, outputs, summary = self._training_pass()
        last_pass_test = False

      # Get the current global step
      step = self.sess.run(self.graph_nodes['global_step'])

      # Display current iteration results
      if step % summary_freq == 0:
        print("|---Done---+---Step---+--Training Loss--+--Sec/Batch--|")
        if FLAGS.write_logs:
          self.writer.add_summary(summary, step)
      if step % (summary_freq / 10) == 0:
        time_taken = time.time() - loop_start_time
        percent_done = 100. * step / Constants.config['total_steps']
        print("|  {:6.2f}%".format(percent_done) + \
              " | {:8d}".format(int(step)) + \
              " | {:.14s}".format("{:14.6f}".format(loss)) + \
              "  | {:.10s}".format("{:10.4f}".format(time_taken)))

      # Save model
      if step > 0 and step % 1000 == 0:
        print("=============== SAVING - DO NOT KILL PROCESS UNTIL COMPLETE ==============")
        self.saver.save(self.sess, self.saver_path)
        print("============================== SAVE COMPLETE =============================")

  def _start_tf_session(self):
    '''
    Additionally set up Tensorboard logging
    '''
    super()._start_tf_session()

    if (FLAGS.write_logs):
      logs_path = os.path.join(self._bin_dir, self._get_name_string())
      self.writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

  def _get_name_string(self):
    dir_string = str(datetime.now().strftime('%m-%d_%H:%M_'))
    dir_string += self.config_file[:self.config_file.rindex('.')]
    return dir_string

def main(argv):
  np.set_printoptions(precision=4, linewidth=150, suppress=True)
  if FLAGS.write_logs:
    print("Tensorboard data will be written for this run")
  else:
    print("Tensorboard data will NOT be written for this run")
    print("Run application with -h for flag usage")

  if '--config_file' not in argv or '--dataset' not in argv or '--source_num_way' not in argv:
    print("Must provide --config_file, --dataset, --source_num_way")
    print("Example:")
    print("tf train_source_models.py --config_file standard_omniglot.yml --dataset omniglot --source_num_way 50")


  config_file = argv[argv.index('--config_file') + 1]
  dataset = argv[argv.index('--dataset') + 1]
  source_num_way = int(argv[argv.index('--source_num_way') + 1])
  random.seed(7218)
  tf.set_random_seed(6459)
  np.random.seed(7518)

  datasets = data_partitioner.load_dataset('datasets', dataset)
  data_interface = DataInterface(datasets)

  num_classes = data_interface.num_classes()

  num_parallel = 8

  splits = []
  for left in np.arange(num_classes - source_num_way):
    right = left + source_num_way
    splits.append(np.arange(left, right))
  indices = np.arange(len(splits))

  print("About to train {} times... that sounds crazy.".format(len(splits)))

  bin_base = os.path.join('bin', 'source_models', "{}_{}".format(dataset, source_num_way))
  if not os.path.exists(bin_base):
    os.makedirs(bin_base)

  with open(os.path.join(bin_base, 'idx_splits.csv'), 'w') as csv_file:
    writer = csv.writer(csv_file)
    for index, split in zip(indices, splits):
      writer.writerow([index, *split])
  def train(index_split):
    index, split = index_split
    bin_dir = os.path.join(bin_base, str(index))
    trainer = SourceTrainer(config_file, description=None, class_indices=split, bin_dir=bin_dir, data_interface=data_interface)
    final_test_score = trainer.run()
    print("Final test score: {:.2f}%".format(100 * final_test_score))
    with open(os.path.join(bin_dir, 'final_score.txt'), 'w') as out_file:
      out_file.write(str(final_test_score))



  pool = ThreadPool(num_parallel)
  pool.map(train, list(zip(indices, splits)))
  print("COMPLETE")

if __name__ == "__main__":
  main(sys.argv)

import collections
import os
import random
import sys
import time
import threading
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

class Trainer(BaseRunner):
  '''
  This is the entry point for training a model. Takes care of loading config file,
  model, loads from file (if saved) etc.
  '''
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.graph_nodes = self.graph_builder.build_graph()

    self._build_datasets()

  def _build_datasets(self):
    '''
    Retrieve the train/test datasets
    '''
    self.datasets = data_partitioner.load_datasets('datasets', Constants.config['dataset'])
    self.train_set = DataInterface(self.datasets['train'], mode='train')
    self.test_set  = DataInterface(self.datasets['test'], mode='test')

  def run(self):
    '''
    Starts the training process - the workhorse
    '''
    start_time = time.time()
    self._start_tf_session()

    self._run_training()

    print("Training complete. \n\tTime taken: {:6.2f} sec".format(time.time() - start_time))
    self.sess.close()
    if FLAGS.write_logs:
      self.writer.close()
    tf.reset_default_graph()
    Helper.reset_tb_data()

  def _test_pass(self, support_set, query_set):
    '''
    A single pass through the given batch from the test set
    Note that _test_pass and _training_pass are separate functions in case their internals diverge.
    '''
    loss, outputs, summary = self.sess.run([
        self.graph_nodes['loss'],
        self.graph_nodes['outputs'],
        self.graph_nodes['test_summary_op']
    ], {
        self.graph_nodes['support_images']: support_set['images'],
        self.graph_nodes['query_images']: query_set['images'],
        self.graph_nodes['input_y']: query_set['labels'],
        self.graph_nodes['is_training']: False
    })
    return loss, outputs, summary

  def _training_pass(self, support_set, query_set):
    '''
    A single pass through the given batch from the training set
    Note that _test_pass and _training_pass are separate functions in case their internals diverge.
    '''
    _, loss, outputs, summary = self.sess.run([
        self.graph_nodes['train_op'],
        self.graph_nodes['loss'],
        self.graph_nodes['outputs'],
        self.graph_nodes['train_summary_op']
    ], {
        self.graph_nodes['support_images']: support_set['images'],
        self.graph_nodes['query_images']: query_set['images'],
        self.graph_nodes['input_y']: query_set['labels'],
        self.graph_nodes['is_training']: True
    })
    return loss, outputs, summary

  def _evaluate(self, outputs, support_set, query_set):
    '''
    Any additional logging you may want.
    '''
    return None

  def _run_training(self):
    '''
    The training/validation loop
    '''
    train_set = self.train_set
    test_set = self.test_set
    summary_freq = 50
    test_pass_freq = 51 # How often to run validation

    step = self.sess.run(self.graph_nodes['global_step'])
    task_str = 'train'
    while step < Constants.config['total_steps']:
      loop_start_time = time.time()

      is_training_pass = step % test_pass_freq != 0 or task_str == 'test'
      task_str = 'train' if is_training_pass else 'test'
      # Run training or test pass
      if is_training_pass:
        support_set, query_set = train_set.get_next_batch()
        loss, outputs, summary = self._training_pass(support_set, query_set)
      else:
        support_set, query_set = test_set.get_next_batch()
        loss, outputs, summary = self._test_pass(support_set, query_set)
      # Get the current global step
      step = self.sess.run(self.graph_nodes['global_step'])

      # Evaluate model's outputs
      tb_stats = self._evaluate(outputs, support_set, query_set)

      # Write logs
      if step % summary_freq == 0 or not is_training_pass:
        if FLAGS.write_logs:
          if tb_stats:
            for key, value in tb_stats.items():
              if value:
                Helper.update_tb_variable(step, task_str, key, value, self.sess, self.writer)
          self.writer.add_summary(summary, step)

      # Display current iteration results
      if step % 30 == 0:
        print("|---Done---+---Step---+--{:>5s}ing Loss--+--Sec/Batch--+---EPS---|".format(task_str))
      time_taken = time.time() - loop_start_time
      eps = float(query_set['labels'].shape[0])/time_taken
      percent_done = 100. * step / Constants.config['total_steps']
      print("|  {:6.2f}%".format(percent_done) + \
            " | {:8d}".format(int(step)) + \
            " | {:.14s}".format("{:14.6f}".format(loss)) + \
            "  | {:.10s}".format("{:10.4f}".format(time_taken)) + \
            "  | {:7.2f}".format(eps) + " |")

      # Save model
      if step % 100 == 0 and is_training_pass:
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
    dir_string = dir_string + '_' + self._description if self._description \
                                                      else dir_string
    return dir_string

def main(argv):
  np.set_printoptions(precision=4, linewidth=150, suppress=True)
  if FLAGS.write_logs:
    print("Tensorboard data will be written for this run")
  else:
    print("Tensorboard data will NOT be written for this run")
    print("Run application with -h for flag usage")

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

  trainer = Trainer(config_file, description, argv)

  trainer.run()

if __name__ == "__main__":
  main(sys.argv)

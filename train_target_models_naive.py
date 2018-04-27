import re
import collections
import os
import random
import sys
import csv
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

class TargetTrainer(BaseRunner):
  '''
  This is the entry point for training a model. Takes care of loading config file,
  model, loads from file (if saved) etc.
  '''
  def __init__(self, config_file, description, source_class_indices, bin_dir, dataset):
    super().__init__(config_file, description=description, argv=None, bin_dir=bin_dir)
    self.graph_nodes, self._model_module = self.graph_builder.build_graph()
    self._source_class_indices = source_class_indices
    datasets = data_partitioner.load_dataset('datasets', dataset)
    self.data_interface =  DataInterface(datasets)

  def _get_new_class_indices(self, source_indices, num_total_classes, num_new_classes):
    to_choose_from = list(set(np.arange(num_total_classes)) - set(source_indices))
    new_class_indices = np.random.choice(to_choose_from, num_new_classes, replace=False)
    return new_class_indices

  def run(self):
    '''
    Starts the training process - the workhorse
    '''
    start_time = time.time()
    self._start_tf_session()
    new_classes = Constants.config['target_num_way'] - Constants.config['num_way']
    num_total_classes = self.data_interface.num_classes()
    self._target_class_indices = self._get_new_class_indices(self._source_class_indices,
                                                             num_total_classes,
                                                             new_classes)
    new_dir_name = 'target_model'
    self.saver_path = os.path.join(self._bin_dir, new_dir_name, 'saver', 'model.ckpt')
    if not os.path.exists(self.saver_path[:self.saver_path.rindex('/')]):
      os.makedirs(self.saver_path[:self.saver_path.rindex('/')])
    self._model_module.prepare_for_training(self.sess, self.graph_nodes)
    self.graph_builder.build_summary_ops()
    self._run_training()

    print("Training complete. \n\tTime taken: {:6.2f} sec".format(time.time() - start_time))
    self.sess.close()
    if FLAGS.write_logs:
      self.writer.close()
    tf.reset_default_graph()
    Helper.reset_tb_data()

  def _training_pass(self):
    '''
    A single pass through the given batch from the training set
    '''
    # Train on TARGET set
    images, labels = self.data_interface.get_next_batch('train', self._target_class_indices, num_shot=Constants.config['num_shot'])
    # Need to offset the labels by the number of source classes, as we want (for 4 classes -> 6 classes):
    # source classes: [0, 1, 2, 3]; target classes = [4, 5]
    labels = np.asarray(labels) + Constants.config['num_way']
    loss, outputs, summary = self._model_module.training_pass(self.sess, self.graph_nodes, self.graph_nodes['target_train_summary_op'], images, labels)
    return loss, outputs, summary

  def _test_pass(self):
    '''
    A single pass through the given batch from the test set
    '''
    # Test TARGET set
    images, labels = self.data_interface.get_next_batch('test', self._target_class_indices, num_shot=Constants.config['num_shot'])
    # Need to offset the labels by the number of source classes, as we want (for 4 classes -> 6 classes):
    # source classes: [0, 1, 2, 3]; target classes = [4, 5]
    labels = np.asarray(labels) + Constants.config['num_way']
    loss_t, outputs_t, summary_t = self._model_module.test_pass(self.sess, self.graph_nodes, self.graph_nodes['target_test_summary_op'], images, labels)
    # Test SOURCE set
    images, labels = self.data_interface.get_next_batch('test', self._source_class_indices, num_shot=Constants.config['num_shot'])
    loss_s, outputs_s, summary_s = self._model_module.test_pass(self.sess, self.graph_nodes, self.graph_nodes['source_test_summary_op'], images, labels)
    return loss_t, outputs_t, summary_t, loss_s, outputs_s, summary_s

  def _run_training(self):
    '''
    The training/validation loop
    '''
    summary_freq = 100

    test_pass_freq = 10 # How often to run validation
    task_str = 'train'
    step = self.sess.run(self.graph_nodes['global_step'])
    while step < Constants.config['total_steps'] + Constants.config['retraining_steps']:
      loop_start_time = time.time()

      is_training_pass = step % test_pass_freq != 0 or task_str == 'test'
      task_str = 'train' if is_training_pass else 'test'
      # Run training or test pass
      if is_training_pass:
        loss, outputs, summary = self._training_pass()
      else:
        loss_t, outputs_t, summary_t, loss_s, outputs_s, summary_s = self._test_pass()
        loss = (loss_t + loss_s) / 2

      # Get the current global step
      step = self.sess.run(self.graph_nodes['global_step'])


      # Write logs
      if FLAGS.write_logs:
        if step % summary_freq == 0 or not is_training_pass:
          if is_training_pass:
            self.writer.add_summary(summary, step)
          else:
            self.writer.add_summary(summary_t, step)
            self.writer.add_summary(summary_s, step)
      # Display current iteration results
      if step % summary_freq == 0:
        print("|---Done---+---Step---+--Training Loss--+--Sec/Batch--|")
      if step % (summary_freq / 10) == 0:
        time_taken = time.time() - loop_start_time
        percent_done = 100. * step / (Constants.config['total_steps'] + Constants.config['retraining_steps'])
        print("|  {:6.2f}%".format(percent_done) + \
              " | {:8d}".format(int(step)) + \
              " | {:.14s}".format("{:14.6f}".format(loss)) + \
              "  | {:.10s}".format("{:10.4f}".format(time_taken)))

      # Save model
      if step % 1000 == 0:
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

  if '--config_file' not in argv or '--dataset' not in argv or '--source_num_way' not in argv or '--target_num_way' not in argv:
    print("Must provide --config_file, --dataset, --target_num_way")
    print("Example:")
    print("tf train_base_models.py --config_file standard_omniglot.yml --dataset omniglot --source_num_way 50 --target_num_way 60")

  config_file = argv[argv.index('--config_file') + 1]
  dataset = argv[argv.index('--dataset') + 1]
  source_num_way = int(argv[argv.index('--source_num_way') + 1])
  target_num_way = int(argv[argv.index('--target_num_way') + 1])
  random.seed(7218)
  tf.set_random_seed(6459)
  np.random.seed(7518)



  num_parallel = 8

  bin_base = os.path.join('bin', 'source_models', '{}_{}'.format(dataset, source_num_way))

  directories = [(os.path.join(bin_base, sub_dir), sub_dir) for sub_dir in os.listdir(bin_base) \
                    if os.path.isdir(os.path.join(bin_base, sub_dir))]

  # List of (directory, split) pairs
  subdir_splits = []
  with open(os.path.join(bin_base, 'idx_splits.csv'), 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
      subdir, *split = row
      subdir_splits.append((subdir, [int(s) for s in split]))

  print("About to train {} times... that sounds crazy.".format(len(subdir_splits)))

  def train(subdir_split):
    subdir, split = subdir_split
    bin_dir = os.path.join(bin_base, subdir)
    trainer = TargetTrainer(config_file, description=None, source_class_indices=split, bin_dir=bin_dir, dataset=dataset)
    trainer.run()


  pool = ThreadPool(num_parallel)
  pool.map(train, subdir_splits)


if __name__ == "__main__":
  main(sys.argv)

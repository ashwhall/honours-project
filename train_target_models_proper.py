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
  def __init__(self, config_file, description, source_bin_base, subdir_splits, bin_dir, dataset, source_num_way, target_num_way):
    super().__init__(config_file, description=description, argv=None, bin_dir=bin_dir)
    self._placeholders_built = False
    self._subdir_splits = subdir_splits
    datasets = data_partitioner.load_dataset('datasets', dataset)
    self.data_interface = DataInterface(datasets)
    self._source_bin_base = source_bin_base
    self._source_num_way = source_num_way
    self._target_num_way = target_num_way
    self._get_grads_weights(0)
    self.graph_nodes, self._model_module = self.graph_builder.build_graph(build_placeholders=True)
    self._num_validation_models = int(len(subdir_splits) / 10.)


  def _get_source_model_things(self, model_index):
    subdir, class_indices = self._subdir_splits[model_index]
    saver_path = './bin/source_models/cifar100_50/{}/saver/model.ckpt.meta'.format(subdir)
    return saver_path, class_indices

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

  def _get_grads_weights(self, model_index):
    # Load old model
    g2 = tf.Graph()
    with g2.as_default():
      with tf.Session(graph=g2) as temp_sess:
        source_saver_path, source_indices = self._get_source_model_things(model_index)
        # Load model from file
        print("Loading from file")

        saver = tf.train.import_meta_graph(source_saver_path)
        saver.restore(temp_sess,tf.train.latest_checkpoint(source_saver_path[:source_saver_path.rindex('/')]))
        from models.meta_learner import MetaLearner
        # Access all trained weights
        print("Evaluating weights")
        all_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        all_weights_np = [temp_sess.run(w) for w in all_weights]

        # Add uninitialized weights for output layer
        print("Updating final layer weights")
        output_weights = all_weights_np[-1]
        new_weights_shape = list(output_weights.shape[:-1]) + [self._target_num_way - self._source_num_way]
        new_vals = temp_sess.run(tf.truncated_normal(new_weights_shape))
        combined_weights = tf.Variable(np.concatenate((output_weights, new_vals), -1), name='new_conv_head')
        all_weights[-1] = combined_weights
        new_old_model = MetaLearner.build_new_model(all_weights)

        # Obtain gradients for target images
        target_indices = self._get_new_class_indices(source_indices, 100, self._target_num_way - self._source_num_way)
        for t in target_indices:
          assert t not in source_indices, "Overlap in source/target indices. Something is wrong!"
        target_images, target_labels = self.data_interface.get_next_batch('train', target_indices, num_shot=Constants.config['num_shot'])
        # print(len(target_labels))
        target_labels = np.asarray(target_labels) + Constants.config['num_way']
        # print(target_labels)
        # print(len(target_labels))
        input_imgs = tf.placeholder(tf.float32, Constants.config['input_shape'])
        predictions = new_old_model(input_imgs)
        # print(predictions)
        targets = tf.placeholder(tf.float32, [None])

        targets_one_hot = tf.one_hot(tf.to_int32(targets), self._target_num_way)

        # print(targets)
        loss = tf.losses.softmax_cross_entropy(targets_one_hot, predictions)
        print("Initializing variables")
        temp_sess.run(tf.global_variables_initializer())
        optimizer = tf.train.GradientDescentOptimizer(1e-3)
        grads_weights = optimizer.compute_gradients(loss)
        grads_weights = [(g, w) for (g, w) in grads_weights if g is not None]

        print("Computing gradients with target images")
        if not self._placeholders_built:
          self.graph_builder.template_grads_weights = grads_weights
        grads_weights = temp_sess.run(grads_weights, {input_imgs: target_images, targets: target_labels})
    return grads_weights, source_indices, target_indices

  def _training_pass(self, model_index):
    '''
    A single pass through the given batch from the training set
    '''
    grads_weights, source_indices, target_indices = self._get_grads_weights(model_index)
    for g, w in grads_weights:
      assert isinstance(g, np.ndarray), "Gradient is not a numpy array"
      assert isinstance(w, np.ndarray), "Weight is not a numpy array"
    print("Performing meta-learner training pass with model {}".format(model_index))
    # Train on TARGET set
    target_images, target_labels = self.data_interface.get_next_batch('test', target_indices, num_shot=Constants.config['num_shot'])
    # Need to offset the labels by the number of source classes, as we want (for 4 classes -> 6 classes):
    # source classes: [0, 1, 2, 3]; target classes = [4, 5]
    target_labels = np.asarray(target_labels) + Constants.config['num_way']
    source_images, source_labels = self.data_interface.get_next_batch('test', source_indices, num_shot=Constants.config['num_shot'])
    images = np.concatenate((target_images, source_images))
    labels = np.concatenate((target_labels, source_labels))
    summaries = []
    print("Combined pass")
    loss, outputs, summary = self._model_module.training_pass(self.sess, self.graph_nodes, self.graph_nodes['combined_train_summary_op'], images, labels, grads_weights)
    summaries.append(summary)
    print("source pass")
    _, _, summary = self._model_module.test_pass(self.sess, self.graph_nodes, self.graph_nodes['source_train_summary_op'], source_images, source_labels, grads_weights)
    summaries.append(summary)
    print("target pass")
    _, _, summary = self._model_module.test_pass(self.sess, self.graph_nodes, self.graph_nodes['target_train_summary_op'], target_images, target_labels, grads_weights)
    summaries.append(summary)
    return loss, outputs, summaries


  def _test_pass(self, model_index):
    '''
    A single pass through the given batch from the training set
    '''
    grads_weights, source_indices, target_indices = self._get_grads_weights(model_index)

    print("Performing meta-learner test pass with model {}".format(model_index))
    # Train on TARGET set
    target_images, target_labels = self.data_interface.get_next_batch('test', target_indices, num_shot=Constants.config['num_shot'])
    # Need to offset the labels by the number of source classes, as we want (for 4 classes -> 6 classes):
    # source classes: [0, 1, 2, 3]; target classes = [4, 5]
    target_labels = np.asarray(target_labels) + Constants.config['num_way']
    source_images, source_labels = self.data_interface.get_next_batch('test', source_indices, num_shot=Constants.config['num_shot'])
    images = np.concatenate((target_images, source_images))
    labels = np.concatenate((target_labels, source_labels))
    summaries = []
    print("Combined pass")
    loss, outputs, summary = self._model_module.test_pass(self.sess, self.graph_nodes, self.graph_nodes['combined_test_summary_op'], images, labels, grads_weights)
    summaries.append(summary)
    print("source pass")
    _, _, summary = self._model_module.test_pass(self.sess, self.graph_nodes, self.graph_nodes['source_test_summary_op'], source_images, source_labels, grads_weights)
    summaries.append(summary)
    print("target pass")
    _, _, summary = self._model_module.test_pass(self.sess, self.graph_nodes, self.graph_nodes['target_test_summary_op'], target_images, target_labels, grads_weights)
    summaries.append(summary)
    return loss, outputs, summaries


  def _run_training(self):
    '''
    The training/validation loop
    '''
    summary_freq = 1

    test_pass_freq = 5 # How often to run validation
    task_str = 'train'
    step = self.sess.run(self.graph_nodes['global_step'])
    while step < Constants.config['meta_learner_training_steps']:
      loop_start_time = time.time()

      is_training_pass = step % test_pass_freq != 0 or task_str == 'test'
      task_str = 'train' if is_training_pass else 'test'
      # Run training or test pass
      losses = []
      if is_training_pass:
        for source_model_index in range(len(self._subdir_splits) - self._num_validation_models):
          loss, outputs, summaries = self._training_pass(source_model_index)
          print("loss: ", loss)
          losses.append(loss)
      else:
        for source_model_index in range(self._num_validation_models):
          # Offset to get past the training models
          source_model_index += len(self._subdir_splits) - self._num_validation_models
          loss, outputs, summaries = self._test_pass(source_model_index)
          print("TEST loss: ", loss)
          losses.append(loss)
      loss = np.mean(losses)
      # Get the current global step
      step = self.sess.run(self.graph_nodes['global_step'])


      # Write logs
      if FLAGS.write_logs:
        print("Writing summaries")
        for summary in summaries:
          self.writer.add_summary(summary, step)
        self.writer.flush()

      # Display current iteration results
      if step % summary_freq == 0:
        print("|---Done---+---Step---+--Training Loss--+--Sec/Batch--|")
      if step % (summary_freq / 10) == 0:
        time_taken = time.time() - loop_start_time
        percent_done = 100. * step / (Constants.config['meta_learner_training_steps'])
        print("|  {:6.2f}%".format(percent_done) + \
              " | {:8d}".format(int(step)) + \
              " | {:.14s}".format("{:14.6f}".format(loss)) + \
              "  | {:.10s}".format("{:10.4f}".format(time_taken)))

      # Save model
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

  print("Training meta-learner over {} models".format(len(subdir_splits)))

  # print(subdir_splits[0])
  bin_dir = os.path.join('bin', 'target_models', 'cifar100_60')
  trainer = TargetTrainer(config_file, description=None, source_bin_base=bin_base, subdir_splits=subdir_splits, bin_dir=bin_dir, dataset=dataset, source_num_way=source_num_way, target_num_way=target_num_way)
  trainer.run()





if __name__ == "__main__":
  main(sys.argv)

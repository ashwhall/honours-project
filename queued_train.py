import sys
import glob
import os
import csv
import shutil
from multiprocessing import Process, Manager, Value
from ctypes import c_char_p
from datetime import datetime
import random

import numpy as np
import tensorflow as tf

from train import Trainer

BASE_DIR = 'experiment_results'
flags = tf.app.flags
FLAGS = flags.FLAGS
FLAGS.write_logs = True

class Experiment:
  '''
  Class which is responsible for running a complete training session, including
  making its required directories and cleaning up after each run.
  '''
  def __init__(self, config_file, description, exp_queue_name):
    self._config_file = config_file
    self._description = description.strip()
    self._exp_queue_name = exp_queue_name
    self._dir_name = self._config_file.rsplit(".", 1)[0]
    self._dir_name = self._dir_name + '_' + self._description if self._description \
                                                              else self._dir_name
    self._tb_log_dir = Manager().Value(c_char_p, '')
    self._make_directories()

  def _make_directories(self):
    '''
    Builds the experiment's required directories
    '''
    experiment_base_dir = os.path.join(BASE_DIR, self._exp_queue_name, self._dir_name)
    _make_dir(experiment_base_dir)
    _make_dir(os.path.join(experiment_base_dir, 'saver'))

  def _run_process(self):
    '''
    The work that a training process does
    '''
    bin_dir = os.path.join(BASE_DIR, self._exp_queue_name, self._dir_name)
    trainer = Trainer(self._config_file, self._description, bin_dir=bin_dir)
    trainer.run()

  def run(self):
    '''
    Runs the training session
    '''
    p = Process(target=self._run_process)
    p.start()
    p.join()

def _read_queue_file(filename):
  '''
  Read a list of pairs of config files and descriptions from file.
  Using a list as we would probably like to preserve order.
  '''
  experiment_info_list = []
  with open(filename, 'r') as f_handle:
    reader = csv.reader(f_handle)
    for row in reader:
      experiment_info_list.append((row[0].strip(), row[1].strip()))
  return experiment_info_list

def _get_exp_queue_name():
  '''
  Builds the experiment name from a timestamp and an optional custom name
  '''
  exp_queue_name = str(datetime.now().strftime('%m-%d_%H:%M'))
  custom_string = input('Enter a name for this experiment, or just press <ENTER>: ')
  custom_string = custom_string.strip()
  exp_queue_name = exp_queue_name + '_' + custom_string if custom_string else exp_queue_name
  return exp_queue_name

def _make_dir(path):
  '''
  Creates a directory if it doesn't yet exist
  '''
  if not os.path.exists(path):
    os.makedirs(path)

def _copy_dir(exp_queue_name, *dir_path):
  '''
  Copies the contents (non-recursively) from the given `dir_path` to the experiment's code dir
  '''
  target_dir = os.path.join(BASE_DIR, exp_queue_name, 'code', *dir_path)
  _make_dir(target_dir)

  # Build a list of files to copy
  files = []
  extensions = ['*.py', '*.yml']
  for ext in extensions:
    files.extend(glob.glob(os.path.join(*dir_path, ext)))

  # Perform the copy operation
  for f in files:
    if os.path.isfile(f):
      shutil.copy(f, target_dir)

def _copy_code(exp_queue_name, queue_file):
  '''
  Creates a copy of the code-base to the appropriate directory
  '''
  # Make the directory to hold the code
  code_dir = os.path.join(BASE_DIR, exp_queue_name, 'code')
  _make_dir(code_dir)

  # Copy the queue file
  shutil.copy(queue_file, os.path.join(BASE_DIR, exp_queue_name))

  # Copy all of the code
  _copy_dir(exp_queue_name, '.')
  _copy_dir(exp_queue_name, 'configs')
  _copy_dir(exp_queue_name, 'data_loader')
  _copy_dir(exp_queue_name, 'data_loader', 'labels')
  _copy_dir(exp_queue_name, 'data_loader', 'samplers')
  _copy_dir(exp_queue_name, 'data_loader', 'samplers')
  _copy_dir(exp_queue_name, 'models')
  _copy_dir(exp_queue_name, 'space_transform')

def main(argv):
  if len(argv) == 2:
    queue_file = argv[1]
  else:
    print("You must provide a 'queue_file' argument")
    print("Example usage:")
    print("\ttf queued_train.py queue.csv")
    print("The file should be comma-separated config file, description pairs - one per line")
    print("Example:")
    print("config_15.yml, with_batchnorm")
    print("config_16.yml, lower_lr")
    print("config_17.yml, higher_lr")
    print("...")
    return

  # Set the random seeds
  random.seed(7218)
  tf.set_random_seed(6459)
  np.random.seed(7518)

  experiment_info_list = _read_queue_file(queue_file)
  exp_queue_name = _get_exp_queue_name()
  experiments = []
  for experiment_info in experiment_info_list:
    experiments.append(Experiment(*experiment_info, exp_queue_name))

  _copy_code(exp_queue_name, queue_file)

  print("Running {} experiments. Results will be saved to {}".format(
      len(experiments),
      os.path.join(BASE_DIR, exp_queue_name)
  ))
  for experiment in experiments:
    experiment.run()
    print("*********************************************************************\n\n\n\n\n\n")

  print("Experiment queue complete")
  print("Results can be found in {}".format(os.path.join(BASE_DIR, exp_queue_name)))

if __name__ == "__main__":
  main(sys.argv)

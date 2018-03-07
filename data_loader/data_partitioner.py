import csv
import importlib
import re
import os

def load_datasets(root_dir, dataset_name):
  import_path = root_dir + '.' + dataset_name.lower() + '.loader'
  dir_path = os.path.join(root_dir, dataset_name.lower())

  dataset_loader = importlib.import_module(import_path, package=None)
  datasets = dataset_loader.load_datasets(dir_path)
  dataset_loader.print_dataset_info(datasets)

  return datasets

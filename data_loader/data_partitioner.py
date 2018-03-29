import csv
import importlib
import re
import os

def load_dataset(root_dir, dataset_name):
  '''
  Load the chosen dataset.
  '''
  import_path = root_dir.replace('/', '.') + '.' + dataset_name.lower() + '.loader'
  dir_path = os.path.join(root_dir, dataset_name.lower())

  data_loader_module = importlib.import_module(import_path)
  data_loader_class = getattr(data_loader_module, 'Loader')
  data_loader = data_loader_class(root_dir)
  data_loader.print_dataset_info()

  return data_loader

import os
import pickle
from PIL import Image

import numpy as np
'''
Quick demonstration of the Omniglot format
'''
def load_datasets(directory):
  '''
  Finds all images and builds their labels
  Labels are designated by the sub-directory in which the images reside.
  Distinct categories exist between train and test.
  '''
  datasets = {}
  for set_name in ['test', 'train']:
    curr_label = 0
    dir_path = os.path.join(directory, set_name)
    paths = list(os.walk(dir_path))
    labels = []
    images = []
    for i, (subdir, _, files) in enumerate(paths):
      print("Reading {} images: {:.2f}%\r".format(set_name, 100 * i / len(paths)), end="", flush=True)
      image_found = False
      for filename in files:
        filename = os.path.join(subdir, filename)
        if '.png' in filename.lower():
          labels.append(curr_label)
          with open(filename, 'rb') as in_file:
            img = Image.open(in_file).resize((28, 28))
            img = np.array(img).astype('float32').reshape((28, 28, 1))
            images.append(img)
          image_found = True
      if image_found:
        curr_label += 1
    datasets[set_name] = {
        'images': np.asarray(images),
        'labels': np.asarray(labels)
    }
  return datasets

def print_dataset_info(the_dict):
  print("Data shape:                          ")
  for set_name in the_dict.keys():
    print(set_name)
    for key in the_dict[set_name].keys():
      print(' ', key, ":", the_dict[set_name][key].shape)

# datasets = load_omniglot_datasets('.')
# print_dataset_info(datasets)

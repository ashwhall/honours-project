import os
import pickle
import numpy as np

'''
Quick demonstration of the CIFAR100 format

Note that unlike usual CIFAR usage, we want to maintain separation between classes
'''

def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict

def load_datasets(directory):
  datasets = {}
  for set_name in ['test', 'train']:
    dataset_dict = unpickle(os.path.join(directory, set_name + '.bin'))
    # Extract relevant info
    images = dataset_dict[b'data']
    labels = dataset_dict[b'fine_labels']
    # Sort by label
    sorteds = sorted(zip(labels, images), key=lambda z: z[0])
    labels, images = zip(*sorteds)
    # Correct pixel order by transposing
    images = np.transpose(np.reshape(images,(-1, 3, 32, 32)), (0, 2, 3, 1))
    images = np.asarray(images)
    labels = np.asarray(labels)

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

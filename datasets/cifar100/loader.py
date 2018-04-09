import os
import pickle
import numpy as np
from constants import Constants
class Loader:
  def __init__(self, directory):
    self._directory = os.path.join(directory, 'cifar100')
    self._datasets = self._load_datasets()

  def num_classes(self):
    '''
    Get count of classes
    '''
    return len(self._datasets['train'].keys())

  def _unpickle(self, file):
    '''
    Read binary files
    '''
    with open(file, 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
    return dict


  def _load_datasets(self):
    '''
    Load images/labels, return dict
    '''
    all_images = []
    all_labels = []
    for set_name in ['test', 'train']:
      dataset_dict = self._unpickle(os.path.join(self._directory, set_name + '.bin'))
      # Extract relevant info
      images = dataset_dict[b'data']
      labels = dataset_dict[b'fine_labels']
      # Sort by label
      sorteds = sorted(zip(labels, images), key=lambda z: z[0])
      labels, images = zip(*sorteds)
      # Correct pixel order by transposing
      images = np.transpose(np.reshape(images,(-1, 3, 32, 32)), (0, 2, 3, 1))
      all_images.extend(images)
      all_labels.extend(labels)

    all_labels = np.asarray(all_labels)
    all_images = np.asarray(all_images)
    sorted_indices = np.argsort(all_labels)
    all_labels = all_labels[sorted_indices]
    all_images = all_images[sorted_indices]

    def train_test_split(imgs, train_pcnt):
      '''
      Splits and returns the given `imgs`. Assumes the same class for all
      '''
      img_count = len(imgs)
      split_index = int(train_pcnt * img_count)
      train_imgs = imgs[:split_index]
      test_imgs = imgs[split_index:]
      return train_imgs, test_imgs

    the_dict = {
        'train': {},
        'test': {}
    }
    train_percent = 0.8
    for label_val in np.arange(np.max(all_labels) + 1):
      label_val_indices = np.where(all_labels == label_val)
      train_imgs, test_imgs = train_test_split(all_images[label_val_indices], train_percent)
      the_dict['train'][label_val] = train_imgs
      the_dict['test'][label_val] = test_imgs

    return the_dict

  def get_next_batch(self, dataset, class_indices, num_shot):
    '''
    Get a batch with labels matching `class_indices`. `num_shot` images per class
    '''
    the_dict = self._datasets[dataset]
    batch_images, batch_labels = [], []
    for class_index in class_indices:
      current_labels = [class_index] * num_shot
      current_images = the_dict[class_index]
      np.random.shuffle(current_images)
      current_images = current_images[:num_shot]
      batch_images.extend(current_images)
      batch_labels.extend(current_labels)
    shuffle_indices = np.random.permutation(len(batch_labels))
    batch_images = np.asarray(batch_images)[shuffle_indices]
    batch_labels = np.asarray(batch_labels)[shuffle_indices]
    # Make the labels zero-based
    _, batch_labels = np.unique(batch_labels, return_inverse=True)
    return batch_images, batch_labels


  def print_dataset_info(self):
    '''
    Inspect the dataset
    '''
    return
    print("Data shape:                          ")
    for set_name, class_indices in self._datasets.items():
      print("{}".format(set_name))
      print(class_indices.keys())
      for class_index, examples in class_indices.items():
        print("  Class {} -> {} examples".format(class_index, len(examples)))

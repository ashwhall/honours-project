import os
import csv
import cv2
import numpy as np
from constants import Constants

'''
Quick demonstration of the miniImageNet format
'''
class Loader:
  def __init__(self, directory):
    # Ignore the directory, as minimagenet is stored elsewhere
    self._csv_directory = '/backup/miniimagenet'
    self._img_directory = os.path.join(self._csv_directory, 'images')

    self._datasets = self._load_csv_lists()

  def num_classes(self):
    return len(self._datasets['train'].keys())

  def _load_img(self, filename):
    img = cv2.imread(os.path.join(self._img_directory, filename))
    img = cv2.resize(img, (256, 256))
    return img

  def get_next_batch(self, dataset, class_indices, num_shot):
    '''
    Get a batch with labels matching `class_indices`. `num_shot` images per class
    '''
    the_dict = self._datasets[dataset]
    batch_files, batch_labels = [], []
    for class_index in class_indices:
      current_labels = [class_index] * num_shot
      curr_files = the_dict[class_index]
      np.random.shuffle(curr_files)
      curr_files = curr_files[:num_shot]
      batch_files.extend(curr_files)
      batch_labels.extend(current_labels)
    shuffle_indices = np.random.permutation(len(batch_labels))
    batch_images = np.asarray([self._load_img(b) for b in np.array(batch_files)[shuffle_indices]])
    batch_labels = np.asarray(batch_labels)[shuffle_indices]
    # Make the labels zero-based
    _, batch_labels = np.unique(batch_labels, return_inverse=True)
    return batch_images, batch_labels

  def _load_csv_list(self, csv_filename):
    filepath = os.path.join(self._csv_directory, '{}.csv'.format(csv_filename))
    with open(filepath, 'r') as f:
      csvreader = csv.reader(f)
      # Skip the header
      next(csvreader)
      filename_list = []
      class_list = []
      class_names = []
      for (filename, classname) in csvreader:
        if classname not in class_names:
          class_names.append(classname)
        filename_list.append(filename)
        class_list.append(class_names.index(classname))
      return filename_list, class_list

  def _load_csv_lists(self):
    def train_test_split(imgs, train_pcnt):
      '''
      Splits and returns the given `imgs`. Assumes the same class for all
      '''
      img_count = len(imgs)
      split_index = int(train_pcnt * img_count)
      train_imgs = imgs[:split_index]
      test_imgs = imgs[split_index:]
      return train_imgs, test_imgs

    all_files = []
    all_labels = []
    # Gather all filenames and labels together
    for set_name in ['train', 'test', 'val']:
      print("Loading {}".format(set_name))
      curr_files, curr_labels = self._load_csv_list(set_name)
      if all_labels:
        curr_labels += np.max(all_labels)+1
      all_files.extend(curr_files)
      all_labels.extend(curr_labels)

      # filename_class[set_name] = all_files, all_labels
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)

    the_dict = {
        'train': {},
        'test': {}
    }
    train_percent = 0.9
    for label_val in np.arange(np.max(all_labels) + 1):
      label_val_indices = np.where(all_labels == label_val)
      train_imgs, test_imgs = train_test_split(all_files[label_val_indices], train_percent)
      the_dict['train'][label_val] = train_imgs
      the_dict['test'][label_val] = test_imgs
    return the_dict


  def print_dataset_info(self):
    '''
    Inspect the dataset
    '''
    print("Data shape:                          ")
    for set_name, class_indices in self._datasets.items():
      print("{}".format(set_name))
      for class_index, examples in class_indices.items():
        print("  Class {} -> {} examples".format(class_index, len(examples)))


def main():
  loader = Loader('/backup/miniimagenet')
  loader.print_dataset_info()

if __name__ == "__main__":
  main()

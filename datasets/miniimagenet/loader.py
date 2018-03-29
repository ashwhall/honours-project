import os
import csv
import cv2
import numpy as np
from constants import Constants

'''
Quick demonstration of the miniImageNet format
'''
class Loader:
  def __init__(self, csv_directory):
    self._csv_directory = Constants.config['imagenet_dir']
    self._img_directory = os.path.join(self._csv_directory, 'images')

    self._filename_class = self._load_csv_lists()

  def num_classes(self, dataset):
    return np.max([self._filename_class[dataset][1]]) + 1

  def get_next_batch(self, dataset, indices, num_shot, query_size=1):
    '''
    Builds a support/query batch and returns it
    '''
    if dataset not in self._filename_class:
      raise ValueError("{} not supported".format(dataset))

    # Pull out the desired filenames/labels
    filenames, labels = self._filename_class[dataset]

    # Build list that = 1 where the labels are those passed as `indices`
    labels_in_use = np.in1d(labels, indices).astype(np.uint8)
    # Convert to indices into the label list
    chosen_label_indices = np.where(labels_in_use)[0]
    # Take the filenames and labels that we may choose from (only the given classes)
    candidate_filenames = filenames[chosen_label_indices]
    candidate_labels = labels[chosen_label_indices]

    support_filenames = []
    support_labels = []
    query_filenames = []
    query_labels = []
    # Find which labels we're working with
    uniques = np.unique(candidate_labels)
    for label_val in uniques:
      # Find where the candidate labels is the one we're after
      wheres = np.where(candidate_labels == label_val)[0]
      # Shuffle them to choose out of order
      np.random.shuffle(wheres)
      # Limit to `num_shot`
      support_wheres = wheres[:num_shot]
      # Take the query indices
      query_wheres = wheres[num_shot:num_shot+query_size]
      # Add the support filenames/labels
      support_filenames.extend(candidate_filenames[support_wheres])
      support_labels.extend(candidate_labels[support_wheres])
      # Add the support filenames/labels
      query_filenames.extend(candidate_filenames[query_wheres])
      query_labels.extend(candidate_labels[query_wheres])

    # Limit query filenames/labels to query_size
    rand_indices = np.random.permutation(len(query_labels))[:query_size]
    query_filenames = np.asarray(query_filenames)[rand_indices]
    query_labels = np.asarray(query_labels)[rand_indices]

    # Make sure that the query classes are in the support classes
    for q in query_labels:
      assert q in support_labels


    # Replace the query labels to match the support classes
    the_wheres = []
    for idx, val in enumerate(query_labels):
      the_where = np.where(support_labels == val)[0]
      if the_where.size > 1:
        the_where = the_where[0]
      the_wheres.append(the_where)

    # Convert support labels from whatever their true value is to range [0, len(indices))
    # (same as was done already for the query labels)
    _, support_labels = np.unique(support_labels, return_inverse=True)

    for idx, the_where in enumerate(the_wheres):
      query_labels[idx] = support_labels[the_where]
    def load_img(filename):
      img = cv2.imread(os.path.join(self._img_directory, filename))
      img = cv2.resize(img, (256, 256))
      return img

    support_images = [load_img(f_name) for f_name in support_filenames]
    support_images = np.reshape(support_images,(-1, 256, 256, 3))[:, :, :, ::-1]
    query_images = [load_img(f_name) for f_name in query_filenames]
    query_images = np.reshape(query_images,(-1, 256, 256, 3))[:, :, :, ::-1]

    support_set = {
        'images': support_images,
        'labels': support_labels
    }
    query_set = {
        'images': query_images,
        'labels': query_labels
    }

    return support_set, query_set


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
      return np.array(filename_list), np.array(class_list)

  def _load_csv_lists(self):
    filename_class = {}
    for set_name in ['train', 'test', 'val']:
      print("Loading {}".format(set_name))
      filename_class[set_name] = self._load_csv_list(set_name)
    return filename_class

  def print_dataset_info(self):
    print("Data shape:                          ")
    for set_name in self._filename_class.keys():
      print(set_name)
      print("\tImages: {}".format(len(self._filename_class[set_name][0])))
      print("\tClasses: {}".format(np.max(self._filename_class[set_name][1]) + 1))

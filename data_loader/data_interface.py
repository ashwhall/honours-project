from enum import Enum
import numpy as np
from scipy.ndimage import interpolation as interp

from constants import Constants

# Keywords for indicating the dataset mode
TRAIN = 'train'
TEST = 'test'
EVAL = 'eval'
MODES = [TRAIN, TEST, EVAL]

class DataInterface:
  '''
  The interface for pulling batches of frames. Asynchronously puts batches into
  `batch_queue`
  '''
  def __init__(self, dataset, mode):
    # We use a different number of query images during evaluation (the big test)
    assert mode in MODES, "Mode must be one of {}".format(MODES)
    self._mode = mode
    self._images = dataset['images']
    self._labels = dataset['labels']
    self._num_classes = np.max(self._labels) + 1

    self._num_way = Constants.config['eval_num_way'] if mode == EVAL \
               else Constants.config['num_way']
    self._num_query_imgs = Constants.config['eval_query_imgs'] if mode == EVAL \
                      else Constants.config['num_query_imgs']

  def _augment_images(self, support_set, query_set):
    # Compute angles of rotation in 90deg increments
    support_rotation_identities = np.random.randint(low=0, high=4, size=support_set['images'].shape[0])
    support_rotation_angles = support_rotation_identities * 90

    query_rotation_angles = np.zeros_like(query_set['labels'])
    for i, val in enumerate(support_set['labels']):
      query_rotation_angles[np.where(query_set['labels'] == val)] = support_rotation_angles[i]

    for (imgs, rotations) in [(support_set['images'], support_rotation_angles), (query_set['images'], query_rotation_angles)]:
      for i, (img, rotation) in enumerate(zip(imgs, rotations)):
        interp.rotate(img, rotation, output=imgs[i])
    return support_set, query_set

  def get_next_batch(self, num_way=None):
    '''
    Returns the next batch from the queue
    '''
    num_way = self._num_way if num_way is None else num_way


    # Choose the class labels
    # Filter images and labels to only the chosen ones
    # Randomly take n_shot images from each -> support_set images/labels
    # Randomly take num_query_imgs from each -> query set images/labels
    # Augment if not EVAL




    # TODO: Simplify!
    # Select some class labels - integers in [0, num_classes)
    chosen_class_labels = np.random.choice(self._num_classes, size=num_way, replace=False)
    # Build an n-hot list where the labels are those selected above
    n_hot_chosen_labels = np.in1d(self._labels, chosen_class_labels).astype(np.uint8)
    # Convert from n-hot to indices into the label list
    chosen_label_indices = np.where(n_hot_chosen_labels)[0]
    remaining_images = self._images[chosen_label_indices]
    remaining_labels = self._labels[chosen_label_indices]

    shuffled_indices = np.random.permutation(remaining_images.shape[0])
    remaining_images = remaining_images[shuffled_indices]
    remaining_labels = remaining_labels[shuffled_indices]

    # As we MUST have one images from each class in the support set, find unique class indices
    # and use `unique_indices`, which is the index of the first occurence of each class label
    uniques, unique_indices = np.unique(remaining_labels, return_index=True)

    # Extract the labels we want
    support_indices = unique_indices
    support_labels = remaining_labels[unique_indices]

    # Replace them with a dummy value
    remaining_labels[unique_indices] = -1
    # Leave only the remaining labels
    query_indices = np.where(remaining_labels >= 0)[0]

    query_indices = query_indices[:self._num_query_imgs]
    query_labels = remaining_labels[query_indices]

    for i, val in enumerate(support_labels):
      query_labels[np.where(query_labels == val)] = i
      support_labels[i] = i

    support_set = {
        'images': remaining_images[support_indices],
        'labels': support_labels
    }
    query_set = {
        'images': remaining_images[query_indices],
        'labels': query_labels
    }

    if self._mode != EVAL:
      support_set, query_set = self._augment_images(support_set, query_set)

    return support_set, query_set

import threading
import numpy as np

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
  def __init__(self, dataset):
    # We use a different number of query images during evaluation (the big test)
    self._images = dataset['images']
    self._labels = dataset['labels']

  def num_classes(self):
    return np.max(self._labels) + 1

  def get_next_batch(self, indices, num_shot, query_size=1):
    '''
    Builds a support/query batch and returns it
    '''
    # Build list that = 1 where the labels are those passed as `indices`
    labels_in_use = np.in1d(self._labels, indices).astype(np.uint8)
    # Convert to indices into the label list
    chosen_label_indices = np.where(labels_in_use)[0]
    # Take the images and labels that we may choose from (only the given classes)
    candidate_images = self._images[chosen_label_indices]
    candidate_labels = self._labels[chosen_label_indices]

    support_images = []
    support_labels = []
    query_images = []
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
      # Add the support images/labels
      support_images.extend(candidate_images[support_wheres])
      support_labels.extend(candidate_labels[support_wheres])
      # Add the support images/labels
      query_images.extend(candidate_images[query_wheres])
      query_labels.extend(candidate_labels[query_wheres])

    # Limit query images/labels to query_size
    rand_indices = np.random.permutation(len(support_labels))[:query_size]
    query_images = np.asarray(query_images)[rand_indices]
    query_labels = np.asarray(query_labels)[rand_indices]

    # Make sure that the query classes are in the support classes
    for q in query_labels:
      assert q in support_labels

    # Replace the query labels to match the indices of the support classes
    for idx, val in enumerate(query_labels):
      query_labels[idx] = np.where(support_labels == val)[0]
    # Convert support labels from whatever their true value is to range [0, len(indices))
    # (same as was done already for the query labels)
    _, support_labels = np.unique(support_labels, return_inverse=True)
    
    support_set = {
        'images': support_images,
        'labels': support_labels
    }
    query_set = {
        'images': query_images,
        'labels': query_labels
    }

    return support_set, query_set

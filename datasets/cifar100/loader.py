import os
import pickle
import numpy as np

class Loader:
  def __init__(self, directory):
    self._directory = os.path.join(directory, 'cifar100')
    self._datasets = self._load_datasets()

  def num_classes(self, dataset):
    return np.max([self._datasets[dataset]['labels']]) + 1

  def _unpickle(self, file):
    with open(file, 'rb') as fo:
      dict = pickle.load(fo, encoding='bytes')
    return dict

  def _load_datasets(self):
    datasets = {}
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
      images = np.asarray(images)
      labels = np.asarray(labels)

      datasets[set_name] = {
          'images': np.asarray(images),
          'labels': np.asarray(labels)
      }
    return datasets

  def get_next_batch(self, dataset, indices, num_shot, query_size=1):
    '''
    Builds a support/query batch and returns it
    '''
    if dataset not in self._datasets:
      raise ValueError("{} not supported".format(dataset))

    # Pull out the desired filenames/labels
    images = self._datasets[dataset]['images']
    labels = self._datasets[dataset]['labels']

    # Build list that = 1 where the labels are those passed as `indices`
    labels_in_use = np.in1d(labels, indices).astype(np.uint8)
    # Convert to indices into the label list
    chosen_label_indices = np.where(labels_in_use)[0]
    # Take the filenames and labels that we may choose from (only the given classes)
    candidate_images = images[chosen_label_indices]
    candidate_labels = labels[chosen_label_indices]

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
    rand_indices = np.random.permutation(len(query_labels))[:query_size]
    query_images = np.asarray(query_images)[rand_indices]
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


    support_set = {
        'images': support_images,
        'labels': support_labels
    }
    query_set = {
        'images': query_images,
        'labels': query_labels
    }

    return support_set, query_set

  def print_dataset_info(self):
    print("Data shape:                          ")
    for set_name in self._datasets.keys():
      print(set_name)
      for key in self._datasets[set_name].keys():
        print(' ', key, ":", self._datasets[set_name][key].shape)

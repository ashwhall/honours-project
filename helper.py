import queue
import functools
import importlib
import tensorflow as tf
import numpy as np
import cv2

class Helper:
  '''
  Utility functions that we may want to use anywhere in the application
  '''
  @staticmethod
  def reset_tb_data():
    '''
    We can clear out the Tensorboard side of things between runs, but as this is
    a static method, we need to manual clear out our python variables
    '''
    Helper.tb_nodes = {}

  # We sometimes want to add regular python variables to Tensorboard. This entails some boilerplate
  # code, so we use `tb_nodes`, conjunction with `register_tb_summary` and `update_tb_variable`
  tb_nodes = {}
  @staticmethod
  def register_tb_summary(key, name, dtype='float32'):
    '''
    As we sometimes want to track python variables in Tensorboard, this function
    allows us to abstract that process. We will add a scalar summary referred to
    by `name` of type `dtype` to a group of summaries referred to by `key`.
    This allows us later to update a value in Tensorboard using update_tb_varable
    Example usage:
      Helper.register_tb_summary('train', 'accuracy')
    '''
    tb_nodes = Helper.tb_nodes
    if key not in tb_nodes:
      tb_nodes[key] = {}
    key_scope_dict = tb_nodes[key]
    # We don't want to register the same key, name combination
    msg = name + " already registered within " + key + " scope."
    assert name not in key_scope_dict, msg
    key_scope_dict[name] = {}
    this_dict = key_scope_dict[name]
    this_dict['tb'] = tf.Variable(0.0, name + '_' + key)
    this_dict['py'] = tf.placeholder(dtype, [])
    this_dict['update'] = this_dict['tb'].assign(this_dict['py'])
    this_dict['summary'] = tf.summary.scalar(name + '_' + key, this_dict['update'])

  @staticmethod
  def update_tb_variable(step, key, name, value, sess, writer):
    '''
    Allows us to update a value in Tensorboard without having to deal with tf
    graph operations. This will run the summary operation, and write it to file
    with the given `sess` and `writer` objects. Argument `step` must be a regular
    Python variable, not a Tensorflow node.
    Example usage:
      Helper.update_tb_variable(global_step,
                                'train',
                                'accuracy',
                                acc_val,
                                sess,
                                writer)
    '''
    this_dict = Helper.tb_nodes[key][name]
    _, summary = sess.run([this_dict['update'], this_dict['summary']],
                          feed_dict={this_dict['py']: value})
    writer.add_summary(summary, step)


  @staticmethod
  def linear_interpolate(x1, x2, y1, y2, x3):
    '''
    Finds the linearly interpolated value for y3 given x3, between the two
    points (x1, y1) and (x2, y2)
    '''
    m = float(y1 - y2) / (x1 - x2)
    y3 = (x3 - x2) * m + y2
    return y3

  @staticmethod
  def detect_blobs(image):
    '''
    Extract coordinates from the given `blob` image.
    '''
    # Convert to uint8
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)

    # Switch out the function called her to change the blob-detection method
    return Helper._cv2_blobdetector(image)

  @staticmethod
  def _watershed(image):
    '''
    Watershed blob detection, taken from http://www.pyimagesearch.com/2015/11/02/watershed-opencv/
    '''
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    coords = []
    # loop over the contours
    for (_, c) in enumerate(cnts):
      ((x, y), _) = cv2.minEnclosingCircle(c)
      coords.append([x, y])
    return coords

  @staticmethod
  def _cv2_blobdetector(image):
    '''
    Blob detection performed by OpenCV's built-in blob-detector
    '''
    # Set up the detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(image)
    return [[k.pt[0], k.pt[1]] for k in keypoints]

  @staticmethod
  def filter_duplicate_coords(coords):
    '''
    Given a list of coords, returns a new list with the average for any
    lanes with more than one coordinate point.
    '''
    result = []

    for lane in range(10):
      temp = []
      for i, _ in enumerate(coords):
        if coords[i][0] == lane:
          temp.append(coords[i][1])
      length = len(temp)
      if length == 1:
        result.append([lane, temp[0]])
      elif length == 0:
        continue
      else:
        result.append([lane, functools.reduce(lambda a, b: a+b, temp)/length])
    return result

  @staticmethod
  def compute_median_world_distance(pred_coords, target_coords):
    '''
    Computes the median distance (in metres) between the given (world-space) `pred_coords`
    and `target_coords`.
    '''
    dists = []
    for pred in pred_coords:
      if pred is None:
        continue
      for target in target_coords:
        if target is None:
          continue
        if pred[0] == target[0]:
          dists.append(abs(pred[1] - target[1]))
    dist_count = len(dists)
    dists = sorted(dists)

    if dist_count > 0:
      median_distance = dists[dist_count // 2] if dist_count % 2 != 0 else \
                        (dists[dist_count // 2 - 1] + dists[dist_count//2]) / 2
    else:
      median_distance = None

    return median_distance

  @staticmethod
  def compute_mean_world_distance(pred_coords, target_coords):
    '''
    Computes the mean distance (in metres) between the given (world-space) `pred_coords`
    and `target_coords`.
    Returns with None if none of the pred_coords are in the same lanes as
    the target_coords
    '''
    dist_sum = 0
    dist_count = 0
    for pred in pred_coords:
      if pred is None:
        continue
      for target in target_coords:
        if target is None:
          continue
        if pred[0] == target[0]:
          dist_sum += abs(pred[1] - target[1])
          dist_count += 1
    mean_distance = dist_sum / dist_count if dist_count != 0 else None
    return mean_distance

  @staticmethod
  def print_lanes(pred_coords, target_coords):
    '''
    Prints the target and predicted coordinates to screen
    '''
    for lane in range(10):
      print("{:2d} ".format(lane), end="")

      found = False
      for coord in target_coords:
        if coord is not None and coord[0] == lane:
          print("{:>5.2f}m ".format(coord[1]), end="")
          found = True
          break
      if not found:
        print(" ----- ", end="")
      found = False
      for coord in pred_coords:
        if coord is not None and coord[0] == lane:
          print("| {:>5.2f}m ".format(coord[1]), end="")
          found = True
          break
      if not found:
        print("| ----- ", end="")
      print()

  @staticmethod
  def compute_recall(pred_coords, target_coords):
    '''
    Computes the recall between the given (world-space) `pred_coords` and `target_coords`.)
    Essentially calculates how many lanes they have in common. Recall may be > 100%, and this is
    worth knowing.
    Returns None if there are no target coords.
    '''
    target_lanes = [coord[0] for coord in target_coords if coord is not None]
    target_count = len(target_lanes)
    pred_count = 0
    for coord in pred_coords:
      if coord[0] in target_lanes:
        pred_count += 1
    return pred_count / target_count if target_count > 0 else None

  @staticmethod
  def class_to_filename(name):
    '''
    Transforms a class name to a file name (without extension).
    E.g. MyCustomClass -> my_custom_class
    '''
    filename = ''
    for letter in name:
      filename += letter if letter.islower() else \
                  '_' + letter.lower()
    filename = filename[1:] if filename[0] == '_' else filename
    return filename

  @staticmethod
  def import_label_class(class_str):
    '''
    Uses the class name as a string to import the class for the label from the
    data_loader folder
    '''
    filename = Helper.class_to_filename(class_str)

    label_module = importlib.import_module('data_loader.labels.' + filename)
    model_class = getattr(label_module, class_str)

    return model_class

  @staticmethod
  def import_video_sampler_class(class_str):
    '''
    Uses the class name as a string to import the class for the data_sampler from the
    data_loader folder
    '''
    filename = Helper.class_to_filename(class_str)

    sampler_module = importlib.import_module('data_loader.samplers.video.' + filename)
    model_class = getattr(sampler_module, class_str)

    return model_class

  @staticmethod
  def import_global_sampler_class(class_str):
    '''
    Uses the class name as a string to import the class for the data_sampler from the
    data_loader folder
    '''
    filename = Helper.class_to_filename(class_str)

    sampler_module = importlib.import_module('data_loader.samplers.global.' + filename)
    model_class = getattr(sampler_module, class_str)

    return model_class

  @staticmethod
  def densify_sparse_targets(sparse_targets):
    '''
    `sparse_targets` is of the form: [ [target00, target01, ...], [target10, ...], ...]
    Where each element is an arbitrary length list of targets.
    Each of the targets must be a simple list with the same number of elements.
      e.g. each target is a co-ordinate in 2D space (2 elements)
    Creates a list that can be stored as a dense numpy array by taking appending
    the index to each of the targets and flattening one dimension.
    returns a list like:
      [ [0] + target00, [0] + target01, ..., [1] + target10, ... ]
    '''
    all_targets = np.empty([0, 3])
    for batch_index, targets in enumerate(sparse_targets):
      if not isinstance(targets, np.ndarray):
        targets = np.array(targets)
      if len(targets) != 0:
        batch_index_column = np.full((targets.shape[0], 1), batch_index)
        targets = np.concatenate([batch_index_column, targets], axis=1)
        all_targets = np.concatenate([all_targets, targets])
    return all_targets

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
  def import_label_class(class_str=None, filename=None):
    '''
    Uses the class name as a string to import the class for the label from the
    data_loader folder
    '''
    if class_str is not None:
      filename = Helper.class_to_filename(class_str)

    label_module = importlib.import_module(filename)
    model_class = getattr(label_module, class_str)

    return model_class

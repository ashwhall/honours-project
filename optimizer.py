import tensorflow as tf
from constants import Constants

class Optimizer:

  @staticmethod
  def build_ops():
    # Ensure the moving averages for the BatchNorm modules are updated.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
      if Constants.config['optimizer'] == 'adam':
        return tf.train.AdamOptimizer(learning_rate=Constants.config['learning_rate'],
                                      epsilon=1e-06)
      elif Constants.config['optimizer'] == 'adadelta':
        return tf.train.AdadeltaOptimizer(learning_rate=Constants.config['learning_rate'])
      elif Constants.config['optimizer'] == 'sgd':
        return tf.train.MomentumOptimizer(learning_rate=Constants.config['learning_rate'], momentum=Constants.config['momentum'])
      else:
        msg = "Optimizer \"" + Constants.config['optimizer'] + "\" not supported."
        raise ValueError(msg)

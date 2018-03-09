import tensorflow as tf

from constants import Constants

from models.relation_network import RelationNetwork

class ReptRelationNetwork(RelationNetwork):
  '''
  The same as relation network, but with reptile optimization
  '''
  def __init__(self, name='ReptileRelationNetwork'):
    super().__init__(name)
    self._model_state = VariableState(tf.trainable_variables())
    self._num_inner_steps = Constants.config['num_inner_steps']
    self._inner_learning_rate = Constants.config['inner_learning_rate']

  def training_pass(self, sess, graph_nodes, support_set, query_set):
    '''
    A single pass through the given batch from the training set
    '''
    old_vars = self._model_state.export_variables(sess)

    for step in range(self._num_inner_steps):
      _, loss, outputs, summary = sess.run([
          graph_nodes['train_op'],
          graph_nodes['loss'],
          graph_nodes['outputs'],
          graph_nodes['train_summary_op']
      ], {
          graph_nodes['support_images']: support_set['images'],
          graph_nodes['query_images']: query_set['images'],
          graph_nodes['input_y']: query_set['labels'],
          graph_nodes['is_training']: True
      })
    new_vars = self._model_state.export_variables(sess)
    self._model_state.import_variables(sess, old_vars)
    self._model_state.import_variables(sess, interpolate_vars(old_vars, new_vars, self._inner_learning_rate))

    return loss, outputs, summary

  def test_pass(self, sess, graph_nodes, support_set, query_set):
    '''
    A single pass through the given batch from the training set
    '''
    loss, outputs, summary = sess.run([
        graph_nodes['loss'],
        graph_nodes['outputs'],
        graph_nodes['test_summary_op']
    ], {
        graph_nodes['support_images']: support_set['images'],
        graph_nodes['query_images']: query_set['images'],
        graph_nodes['input_y']: query_set['labels'],
        graph_nodes['is_training']: False
    })
    return loss, outputs, summary

class VariableState:
  """
  Manage the state of a set of variables.
  """
  def __init__(self, variables):
    self._variables = variables
    self._placeholders = [tf.placeholder(v.dtype.base_dtype, shape=v.get_shape())
                          for v in variables]
    assigns = [tf.assign(v, p) for v, p in zip(self._variables, self._placeholders)]
    self._assign_op = tf.group(*assigns)

  def export_variables(self, sess):
    """
    Save the current variables.
    """
    return sess.run(self._variables)

  def import_variables(self, sess, values):
    """
    Restore the variables.
    """
    sess.run(self._assign_op, feed_dict=dict(zip(self._placeholders, values)))

import numpy as np
import tensorflow as tf

def interpolate_vars(old_vars, new_vars, epsilon):
  """
  Interpolate between two sequences of variables.
  """
  return add_vars(old_vars, scale_vars(subtract_vars(new_vars, old_vars), epsilon))

def average_vars(var_seqs):
  """
  Average a sequence of variable sequences.
  """
  res = []
  for variables in zip(*var_seqs):
    res.append(np.mean(variables, axis=0))
  return res

def subtract_vars(var_seq_1, var_seq_2):
  """
  Subtract one variable sequence from another.
  """
  return [v1 - v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def add_vars(var_seq_1, var_seq_2):
  """
  Add two variable sequences.
  """
  return [v1 + v2 for v1, v2 in zip(var_seq_1, var_seq_2)]

def scale_vars(var_seq, scale):
  """
  Scale a variable sequence.
  """
  return [v * scale for v in var_seq]

def weight_decay(rate, variables=None):
  """
  Create an Op that performs weight decay.
  """
  if variables is None:
    variables = tf.trainable_variables()
  ops = [tf.assign(var, var * rate) for var in variables]
  return tf.group(*ops)

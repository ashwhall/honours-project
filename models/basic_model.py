import sonnet as snt
import tensorflow as tf
import numpy as np

from models.base_model import BaseModel
import models.layers as Layers

class BasicModel(BaseModel):
  def __init__(self, name='BasicModel'):
    super().__init__(name=name)


  TARGET_SHAPE = [None]
  INPUT_SHAPE = [None, 105, 105, 3]
  def _build(self, inputs, graph_nodes): # pylint: disable=W0221
    '''
    Abstract method - build the Sonnet module.

    Args:
        inputs (tf.Tensor):
        graph_nodes (dict{string->tf.Tensor}): Hooks to common tensors

    Returns:
        outputs (arbitrary structure of tf.Tensors)
    '''
    is_training = graph_nodes['is_training']

    inputs = Layers.conv2d(output_channels=16)(inputs)
    inputs = Layers.max_pool(inputs)
    inputs = snt.BatchNorm()(inputs, is_training=is_training)
    inputs = tf.nn.relu(inputs)

    inputs = Layers.conv2d(output_channels=32)(inputs)
    inputs = Layers.max_pool(inputs)
    inputs = snt.BatchNorm()(inputs, is_training=is_training)
    inputs = tf.nn.relu(inputs)

    inputs = Layers.conv2d(output_channels=64)(inputs)
    inputs = Layers.max_pool(inputs)
    inputs = snt.BatchNorm()(inputs, is_training=is_training)
    inputs = tf.nn.relu(inputs)

    inputs = Layers.conv2d(output_channels=64)(inputs)
    inputs = Layers.max_pool(inputs)
    inputs = snt.BatchNorm()(inputs, is_training=is_training)
    inputs = tf.nn.relu(inputs)

    inputs = snt.BatchFlatten()(inputs)
    inputs = snt.Linear(50)(inputs)
    inputs = tf.nn.relu(inputs)
    inputs = snt.Linear(5)(inputs)

    self.outputs = inputs
    return self.outputs

  def get_loss(self, graph_nodes):
    '''
    Build and return the loss calculation ops. Assume that graph_nodes contains the nodes you need,
    as a KeyError will be raised if a key is missing.
    '''
    targets = graph_nodes['input_y']
    targets = tf.one_hot(tf.to_int32(targets), 5)
    return tf.losses.softmax_cross_entropy(targets, self.outputs)

  def get_target_tensors(self):
    '''
    Returns an arbitrarily nested structure of tensors that are the required input for
    calculating the loss.
    '''
    return tf.placeholder(tf.float32, shape=self.TARGET_SHAPE, name="input_y")

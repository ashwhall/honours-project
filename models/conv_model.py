import sonnet as snt
import tensorflow as tf
import numpy as np

from models.base_model import BaseModel
import models.layers as Layers
from constants import Constants

class ConvModel(BaseModel):
  def __init__(self, name='StandardModel'):
    super().__init__(name=name)

  def _build(self, support_images, graph_nodes): # pylint: disable=W0221
    is_training = graph_nodes['is_training']

    inputs = Layers.conv2d(output_channels=16, use_bias=False)(support_images)
    inputs = Layers.max_pool(inputs)
    # inputs = snt.BatchNorm()(inputs, is_training=is_training)
    inputs = tf.nn.relu(inputs)

    inputs = Layers.conv2d(output_channels=32, use_bias=False)(inputs)
    inputs = Layers.max_pool(inputs)
    # inputs = snt.BatchNorm()(inputs, is_training=is_training)
    inputs = tf.nn.relu(inputs)

    inputs = Layers.conv2d(output_channels=64, use_bias=False)(inputs)
    inputs = Layers.max_pool(inputs)
    # inputs = snt.BatchNorm()(inputs, is_training=is_training)
    inputs = tf.nn.relu(inputs)

    inputs = Layers.conv2d(output_channels=64, use_bias=False)(inputs)
    inputs = Layers.max_pool(inputs)
    # inputs = snt.BatchNorm()(inputs, is_training=is_training)
    inputs = tf.nn.relu(inputs)
    # print(inputs.shape)
    num_classes = Constants.config['num_way']
    inputs = Layers.conv2d(output_channels=num_classes, use_bias=False)(inputs)
    inputs = Layers.global_pool(inputs)
    inputs = tf.reshape(inputs, [-1, num_classes])
    self.outputs = inputs
    return self.outputs

  def get_loss(self, graph_nodes):
    '''
    Build and return the loss calculation ops. Assume that graph_nodes contains the nodes you need,
    as a KeyError will be raised if a key is missing.
    '''
    targets = graph_nodes['labels']
    targets = tf.one_hot(tf.to_int32(targets), Constants.config['num_way'])
    return tf.losses.softmax_cross_entropy(targets, self.outputs)

  def get_target_tensors(self):
    '''
    Returns an arbitrarily nested structure of tensors that are the required input for
    calculating the loss.
    '''
    return tf.placeholder(tf.float32, shape=self.TARGET_SHAPE, name="input_y")

  def training_pass(self, sess, graph_nodes, summary_op, images, labels):
    '''
    A single pass through the given batch from the training set
    '''
    _, loss, outputs, summary = sess.run([
        graph_nodes['train_op'],
        graph_nodes['loss'],
        graph_nodes['outputs'],
        summary_op
    ], {
        graph_nodes['images']: images,
        graph_nodes['labels']: labels,
        graph_nodes['is_training']: True
    })
    return loss, outputs, summary

  def test_pass(self, sess, graph_nodes, summary_op, images, labels):
    '''
    A single pass through the given batch from the training set
    '''
    loss, outputs, summary = sess.run([
        graph_nodes['loss'],
        graph_nodes['outputs'],
        summary_op
    ], {
        graph_nodes['images']: images,
        graph_nodes['labels']: labels,
        graph_nodes['is_training']: False
    })
    return loss, outputs, summary

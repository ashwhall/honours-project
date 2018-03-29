import sonnet as snt
import tensorflow as tf
import numpy as np

from models.base_model import BaseModel
import models.layers as Layers
from constants import Constants

class StandardModel(BaseModel):
  def __init__(self, name='StandardModel'):
    super().__init__(name=name)

  def _build(self, support_images, query_images, graph_nodes): # pylint: disable=W0221
    is_training = graph_nodes['is_training']

    inputs = Layers.conv2d(output_channels=16)(support_images)
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
    inputs = snt.Linear(Constants.config['num_way'], name='class_linear')(inputs)

    self.outputs = inputs
    return self.outputs

  def get_loss(self, graph_nodes):
    '''
    Build and return the loss calculation ops. Assume that graph_nodes contains the nodes you need,
    as a KeyError will be raised if a key is missing.
    '''
    targets = graph_nodes['input_y']
    targets = tf.one_hot(tf.to_int32(targets), Constants.config['num_way'])
    return tf.losses.softmax_cross_entropy(targets, self.outputs)

  def get_target_tensors(self):
    '''
    Returns an arbitrarily nested structure of tensors that are the required input for
    calculating the loss.
    '''
    return tf.placeholder(tf.float32, shape=self.TARGET_SHAPE, name="input_y")

  def _get_class_indices(self, data_interface, dataset, num_way):
    '''
    Builds and returns a list of indices for the classes we wish to sample (always the same classes)
    '''
    chosen_class_labels = np.arange(num_way)
    return chosen_class_labels

  def training_pass(self, sess, graph_nodes, data_interface, class_indices):
    '''
    A single pass through the given batch from the training set
    '''
    support_set, query_set = data_interface.get_next_batch('train', class_indices, num_shot=Constants.config['num_shot'])
    _, loss, outputs, summary = sess.run([
        graph_nodes['train_op'],
        graph_nodes['loss'],
        graph_nodes['outputs'],
        graph_nodes['train_summary_op']
    ], {
        graph_nodes['support_images']: support_set['images'],
        graph_nodes['input_y']: support_set['labels'],
        graph_nodes['is_training']: True
    })
    return loss, outputs, summary

  def test_pass(self, sess, graph_nodes, data_interface, class_indices):
    '''
    A single pass through the given batch from the training set
    '''
    support_set, query_set = data_interface.get_next_batch('train', class_indices, num_shot=Constants.config['num_shot'])
    loss, outputs, summary = sess.run([
        graph_nodes['loss'],
        graph_nodes['outputs'],
        graph_nodes['test_summary_op']
    ], {
        graph_nodes['support_images']: support_set['images'],
        graph_nodes['input_y']: support_set['labels'],
        graph_nodes['is_training']: False
    })
    return loss, outputs, summary

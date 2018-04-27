import sonnet as snt
import tensorflow as tf
import numpy as np

from models.standard_model import StandardModel
import models.layers as Layers
from constants import Constants

class ExtendModelFreezeButNewHead(StandardModel):
  '''
  The model name and the build function must be the same (in terms of what tensorflow sees and name scopes)
  '''
  def __init__(self, name='StandardModel'):
    super().__init__(name=name)

  def _build(self, support_images, graph_nodes): # pylint: disable=W0221
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
    self._output_layer_inputs = tf.nn.relu(inputs)

    self._output_layer = snt.Linear(Constants.config['num_way'], name='class_linear')
    self._output_layer_outputs = self._output_layer(inputs)
    return self._output_layer_outputs

  def prepare_for_training(self, sess, graph_nodes):
    '''
    Adds weights/biases to the output layer so its number of outputs matches source num_way + target num_way
    and replaces 'outputs' in `graph_nodes`
    Also replaces graph_nodes['train_op'] with some frozen weights
    '''
    ##### EXTEND LAYER #####
    # Get the current weight values
    weights_tf, biases_tf = self._output_layer.get_variables()
    weights, biases = sess.run([weights_tf, biases_tf])
    # Compute number of new weights to be added
    num_new_outputs = Constants.config['target_num_way'] - biases.shape[0]

    # Create, connect and initialise the new layer
    new_layer = snt.Linear(num_new_outputs, name='class_linear2')
    new_layer_outputs = new_layer(self._output_layer_inputs)
    sess.run(tf.variables_initializer(new_layer.get_variables()))
    self._output_layer_outputs = tf.concat([self._output_layer_outputs, new_layer_outputs], -1)

    # Initialise with the new values
    # Replace the model class layer outputs
    graph_nodes['outputs'] = self._output_layer_outputs
    graph_nodes['loss'] = self.get_loss(graph_nodes, num_way=Constants.config['target_num_way'])
    ##### FREEZE WEIGHTS #####
    # Get all (gradient, weight) pairs
    grads = graph_nodes['optimizer'].compute_gradients(graph_nodes['loss'])
    # Only keep the gradients for the added FC layer
    allowed_gradients = [(grad, weight) for (grad, weight) in grads if weight in new_layer.get_variables()]
    # Replace the train op with our limited update
    graph_nodes['train_op'] = graph_nodes['optimizer'].apply_gradients(allowed_gradients, global_step=graph_nodes['global_step'])

    # Initialise the variables created by the optimizer
    leftover_strings = set([v.decode('UTF-8') for v in sess.run(tf.report_uninitialized_variables())])
    leftover_vars = [v for v in tf.global_variables() if v.name.split(':')[0] in leftover_strings]
    sess.run(tf.variables_initializer(leftover_vars))


  def get_loss(self, graph_nodes, num_way=None):
    '''
    Build and return the loss calculation ops. Assume that graph_nodes contains the nodes you need,
    as a KeyError will be raised if a key is missing.
    '''
    num_way = Constants.config['num_way'] if num_way is None else num_way
    targets = graph_nodes['labels']
    targets = tf.one_hot(tf.to_int32(targets), num_way)
    return tf.losses.softmax_cross_entropy(targets, self._output_layer_outputs)

  def get_target_tensors(self):
    '''
    Returns an arbitrarily nested structure of tensors that are the required input for
    calculating the loss.
    '''
    return tf.placeholder(tf.float32, shape=self.TARGET_SHAPE, name="labels")

  def _get_class_indices(self, dataset, num_way):
    '''
    Builds and returns a list of indices for the classes we wish to sample (always the same classes)
    '''
    chosen_class_labels = np.arange(num_way)
    return chosen_class_labels


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

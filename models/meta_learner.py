import sonnet as snt
import tensorflow as tf
import numpy as np

from models.base_model import BaseModel
import models.layers as Layers
from constants import Constants
from models.embedder import Embedder
from models.encoder import Encoder

class MetaLearner(BaseModel):
  def __init__(self, name='MetaLearner'):
    super().__init__(name=name)
    self._placeholders = []

  def _strip_name(self, name):
    name_list = name.replace(':', ' ').replace('/', ' ').split()
    if len(name_list) > 2:
      return name_list[1]
    return name_list[0]


  def build_placeholders(self, source_num_way, target_num_way,  grads_weights):
    '''
      Given a list of pairs of (gradient, weight) tensors, builds and returns a list placeholders of the same shape
      '''
    self._source_num_way = source_num_way
    self._target_num_way = target_num_way
    for g, w in grads_weights:
      name = self._strip_name(w.name)
      gradient_ph = tf.placeholder(tf.float32, g.shape,   name=name + '_grad_placeholder')
      weight_ph = tf.placeholder(tf.float32, w.shape, name=name + '_placeholder')
      self._placeholders.append((gradient_ph, weight_ph))
    total = 0
    for grad, weight in self._placeholders:
      total += np.prod(grad.shape.as_list())
      total += np.prod(weight.shape.as_list())
    print("Number of parameters:", total)
    return self._placeholders

  def _build(self, images, graph_nodes): # pylint: disable=W0221
    embedder = Embedder()
    embedded_grads_weights = embedder.embed_all_grads_weights(self._placeholders)
    # Fake batching
    embedded_grads_weights = tf.expand_dims(embedded_grads_weights, 0)
    encoder = Encoder(self._source_num_way, self._target_num_way)
    encoded = encoder.encode(embedded_grads_weights)
    decoded = encoder.decode(encoded)
    # Fake batching
    decoded = tf.squeeze(decoded, [0])
    weight_updates = embedder.unembed_all_weights(decoded)

    # Get the updated model
    model_forward = self._build_model_from_placeholders_updates(weight_updates)
    self.outputs = model_forward(images)
    return self.outputs


  def _build_model_from_placeholders_updates(self, weight_updates):
    return MetaLearner.build_new_model([self._placeholders[0][1] + weight_updates[0],
                                        self._placeholders[1][1] + weight_updates[1],
                                        self._placeholders[2][1] + weight_updates[2],
                                        self._placeholders[3][1] + weight_updates[3],
                                        self._placeholders[4][1] + weight_updates[4]])

  @staticmethod
  def build_new_model(weights):
    def model_forward(inputs):
      nonlocal weights
      # Create tf.Variables if required
      weights = [tf.Variable(w) if isinstance(w, np.ndarray) else w for w in weights]

      outputs = tf.nn.conv2d(inputs, weights[0], [1, 1, 1, 1], padding='SAME', name='new_conv1')
      outputs = Layers.max_pool(outputs)
      outputs = tf.nn.relu(outputs)

      outputs = tf.nn.conv2d(outputs, weights[1], [1, 1, 1, 1], padding='SAME', name='new_conv2')
      outputs = Layers.max_pool(outputs)
      outputs = tf.nn.relu(outputs)

      outputs = tf.nn.conv2d(outputs, weights[2], [1, 1, 1, 1], padding='SAME', name='new_conv3')
      outputs = Layers.max_pool(outputs)
      outputs = tf.nn.relu(outputs)

      outputs = tf.nn.conv2d(outputs, weights[3], [1, 1, 1, 1], padding='SAME', name='new_conv4')
      outputs = Layers.max_pool(outputs)
      outputs = tf.nn.relu(outputs)

      outputs = tf.nn.conv2d(outputs, weights[4], [1, 1, 1, 1], padding='SAME', name='new_conv5')
      outputs = Layers.global_pool(outputs)
      # Reshape to one-hot predictions
      outputs = tf.reshape(outputs, [-1, weights[-1].shape.as_list()[-1]])
      return outputs

    return model_forward

  def get_loss(self, graph_nodes):
    '''
    Build and return the loss calculation ops. Assume that graph_nodes contains the nodes you need,
    as a KeyError will be raised if a key is missing.
    '''
    targets = graph_nodes['labels']
    targets = tf.one_hot(tf.to_int32(targets), self._target_num_way)
    return tf.losses.softmax_cross_entropy(targets, self.outputs)

  def get_target_tensors(self):
    '''
    Returns an arbitrarily nested structure of tensors that are the required input for
    calculating the loss.
    '''
    return tf.placeholder(tf.float32, shape=self.TARGET_SHAPE, name="input_y")

  def training_pass(self, sess, graph_nodes, summary_op, images, labels, grads_weights):
    '''
    A single pass through the given batch from the training set
    '''
    feed_dict = {
      graph_nodes['images']: images,
      graph_nodes['labels']: labels,
      graph_nodes['is_training']: True
    }
    for (grad, weight), (grad_ph, weight_ph) in zip(grads_weights, self._placeholders):
      feed_dict[grad_ph] = grad
      feed_dict[weight_ph] = weight

    _, loss, outputs, summary = sess.run([
      graph_nodes['train_op'],
      graph_nodes['loss'],
      graph_nodes['outputs'],
      summary_op
    ], feed_dict)
    return loss, outputs, summary

  def test_pass(self, sess, graph_nodes, summary_op, images, labels, grads_weights):
    '''
    A single pass through the given batch from the training set
    '''
    feed_dict = {
      graph_nodes['images']: images,
      graph_nodes['labels']: labels,
      graph_nodes['is_training']: False
    }
    for (grad, weight), (grad_ph, weight_ph) in zip(grads_weights, self._placeholders):
      feed_dict[grad_ph] = grad
      feed_dict[weight_ph] = weight

    loss, outputs, summary = sess.run([
      graph_nodes['loss'],
      graph_nodes['outputs'],
      summary_op
    ], feed_dict)
    return loss, outputs, summary

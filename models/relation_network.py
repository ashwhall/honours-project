import sonnet as snt
import tensorflow as tf
import numpy as np

from models.base_model import BaseModel
import models.layers as Layers

def conv_block(inputs, is_training):
  inputs = Layers.conv2d(output_channels=64)(inputs)
  inputs = snt.BatchNorm(update_ops_collection=None)(inputs, is_training=is_training)
  return tf.nn.relu(inputs)

class EmbeddingModule(snt.AbstractModule):
  def __init__(self, name='EmbeddingModule'):
    super().__init__(name=name)

  def _build(self, inputs, is_training):

    inputs = conv_block(inputs, is_training)
    inputs = Layers.max_pool(inputs)

    inputs = conv_block(inputs, is_training)
    inputs = Layers.max_pool(inputs)

    inputs = conv_block(inputs, is_training)

    features = conv_block(inputs, is_training)
    return features

class RelationModule(snt.AbstractModule):
  def __init__(self, name='RelationModule'):
    super().__init__(name=name)

  def _build(self, support_embedding, query_embedding, is_training):
    # Add the batch dimension
    query_embedding = [query_embedding]
    # Tile the query embedding for each of the support images
    query_embedding = tf.tile(query_embedding, [tf.shape(support_embedding)[0], 1, 1, 1])
    # Concatenate the embeddings
    combined_embedding = tf.concat([support_embedding, query_embedding], -1)
    outputs = conv_block(combined_embedding, is_training)
    outputs = Layers.max_pool(outputs)
    outputs = conv_block(outputs, is_training)
    outputs = Layers.max_pool(outputs)

    outputs = snt.BatchFlatten()(outputs)
    outputs = snt.Linear(8)(outputs)
    outputs = tf.nn.relu(outputs)
    outputs = snt.Linear(1)(outputs)
    outputs = tf.nn.sigmoid(outputs)

    outputs = tf.reshape(outputs, [-1])
    return outputs

class RelationNetwork(BaseModel):
  def __init__(self, name='RelationNetwork'):
    super().__init__(name=name)

  def _build(self, support_images, query_images, graph_nodes): # pylint: disable=W0221
    '''
    Args:
        inputs (tf.Tensor):
        graph_nodes (dict{string->tf.Tensor}): Hooks to common tensors

    Returns:
        outputs (arbitrary structure of tf.Tensors)
    '''
    is_training = graph_nodes['is_training']
    support_images = tf.image.resize_images(support_images, [28, 28])
    query_images = tf.image.resize_images(query_images, [28, 28])

    embedding_module = EmbeddingModule()
    support_embedding = embedding_module(support_images, is_training)
    query_embedding = embedding_module(query_images, is_training)

    relation_module = RelationModule()
    relation_func = lambda embedding: relation_module(support_embedding, embedding, is_training)

    # outputs = relation_module(combined_embedding, is_training)
    outputs = tf.map_fn(relation_func, query_embedding)
    self.outputs = outputs
    return self.outputs

  def get_loss(self, graph_nodes):
    '''
    Build and return the loss calculation ops. Assume that graph_nodes contains the nodes you need,
    as a KeyError will be raised if a key is missing.
    '''
    targets = graph_nodes['input_y']
    targets = tf.one_hot(tf.to_int32(targets), tf.shape(graph_nodes['support_images'])[0])
    predictions = self.outputs
    return tf.losses.mean_squared_error(labels=targets, predictions=predictions)

  def get_target_tensors(self):
    '''
    Returns an arbitrarily nested structure of tensors that are the required input for
    calculating the loss.
    '''
    return tf.placeholder(tf.float32, shape=self.TARGET_SHAPE, name="input_y")

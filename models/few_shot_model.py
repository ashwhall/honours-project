import abc
import sonnet as snt
import tensorflow as tf
import numpy as np
from constants import Constants
from models.base_model import BaseModel

class FewShotModel(BaseModel):
  '''
  The abstract base class for all models.
  Note that EarlyFusionModel is already a sub-class of this, so extend that if using early fusion.
  '''
  INPUT_SHAPE = Constants.config['input_shape']
  TARGET_SHAPE = [None]

  def __init__(self, name):
    super().__init__(name=name)

  def _get_class_indices(self, dataset, num_way):
    '''
    Builds and returns a list of indices for the classes we wish to sample
    '''
    num_classes = dataset.num_classes()
    chosen_class_labels = np.random.choice(num_classes, size=num_way, replace=False)
    return chosen_class_labels

  def training_pass(self, sess, graph_nodes, train_set):
    '''
    A single pass through the given batch from the training set
    '''
    class_indices = self._get_class_indices(train_set, Constants.config['num_way'])
    support_set, query_set = train_set.get_next_batch(class_indices, num_shot=Constants.config['num_shot'])
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
    return loss, outputs, summary

  def test_pass(self, sess, graph_nodes, test_set):
    '''
    A single pass through the given batch from the training set
    '''
    class_indices = self._get_class_indices(test_set, Constants.config['num_way'])
    support_set, query_set = test_set.get_next_batch(class_indices, num_shot=Constants.config['num_shot'])
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

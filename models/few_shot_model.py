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

  def _get_class_indices(self, data_interface, dataset, num_way):
    '''
    Builds and returns a list of indices for the classes we wish to sample
    '''
    num_classes = data_interface.num_classes(dataset)
    chosen_class_labels = np.random.choice(num_classes, size=num_way, replace=False)
    return chosen_class_labels

  def training_pass(self, sess, graph_nodes, data_interface):
    '''
    A single pass through the given batch from the training set
    '''
    num_way = Constants.config['num_way']
    num_shot = Constants.config['num_shot']
    num_query_imgs = Constants.config['num_query_imgs']
    class_indices = self._get_class_indices(data_interface, 'train', num_way)
    support_set, query_set = data_interface.get_next_batch('train', class_indices, num_shot=num_shot, query_size=num_query_imgs)
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

  def test_pass(self, sess, graph_nodes, data_interface):
    '''
    A single pass through the given batch from the training set
    '''
    num_way = Constants.config['num_way']
    num_shot = Constants.config['num_shot']
    num_query_imgs = Constants.config['num_query_imgs']
    class_indices = self._get_class_indices(data_interface, 'test', num_way)
    support_set, query_set = data_interface.get_next_batch('test', class_indices, num_shot=num_shot, query_size=num_query_imgs)
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

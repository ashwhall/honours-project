import abc
import sonnet as snt
import tensorflow as tf

class BaseModel(snt.AbstractModule):
  '''
  The abstract base class for all models.
  Note that EarlyFusionModel is already a sub-class of this, so extend that if using early fusion.
  '''
  INPUT_SHAPE = None
  TARGET_SHAPE = None

  def __init__(self, name):
    super().__init__(name=name)

  @abc.abstractmethod
  def _build(self, inputs, graph_nodes): # pylint: disable=W0221
    '''
    Abstract method - build the Sonnet module.

    Args:
        inputs (tf.Tensor):
        graph_nodes (dict{string->tf.Tensor}): Hooks to common tensors

    Returns:
        outputs (arbitrary structure of tf.Tensors)
    '''
    pass

  @abc.abstractmethod
  def get_loss(self, graph_nodes):
    '''
    Build and return the loss calculation ops. Assume that graph_nodes contains the nodes you need,
    as a KeyError will be raised if a key is missing.
    '''
    pass

  def get_target_tensors(self):
    '''
    Returns an arbitrarily nested structure of tensors that are the required input for
    calculating the loss.
    '''
    return tf.placeholder(tf.float32, shape=self.TARGET_SHAPE, name="input_y")

  def get_predicted_outputs(self, outputs):
    '''
    Given the outputs from _build(), select which parts are predictions based solely on the input video
    '''
    return outputs

  def separate_outputs(self, outputs):
    '''
    Given a batch of outputs from _build(), separate them into an iterable in which
    each element has a single frames' output.
    '''
    return outputs

  def training_pass(self, sess, graph_nodes, support_set, query_set):
    '''
    A single pass through the given batch from the training set
    '''
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

  def test_pass(self, sess, graph_nodes, support_set, query_set):
    '''
    A single pass through the given batch from the training set
    '''
    loss, outputs, summary = self.sess.run([
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

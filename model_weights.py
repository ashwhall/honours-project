import tensorflow as tf
from models.meta_learner import MetaLearner
from constants import Constants
class MockConstants(Constants):
  def __init__(self):
    self.config = {
      'input_shape': (32, 32, 3)
    }
with tf.Session() as temp_sess:
  #First let's load meta graph and restore weights
  saver = tf.train.import_meta_graph('/home/ash/python/honours-project/bin/source_models/cifar100_50/0/saver/model.ckpt.meta')
  saver.restore(temp_sess,tf.train.latest_checkpoint('/home/ash/python/honours-project/bin/source_models/cifar100_50/0/saver/'))


  # Access saved Variables directly
  all_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  print(MetaLearner.build_new_model(all_weights ))

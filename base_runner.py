import os.path
import tensorflow as tf

from constants import Constants
from graph_builder import GraphBuilder

class BaseRunner:
  '''
  Sets up the graph, loads the model, adds constants from config file to globally accessable
  variable, and has constants parse command line arguments to override config parameters
  '''
  def __init__(self, config_file, description, argv=None, bin_dir=None):
    self.config_file = config_file
    Constants.load_config(config_file)
    Constants.parse_args(argv)
    self._description = description
    self._bin_dir = bin_dir if bin_dir else 'bin'
    saver_path = os.path.join(self._bin_dir, 'saver')
    if not os.path.exists(saver_path):
      os.makedirs(saver_path)

    self.graph_builder = GraphBuilder(Constants.config['model_name'])

  def _start_tf_session(self):
    '''
    Start the session, set up model loading/saving
    '''
    self.saver_path = os.path.join(self._bin_dir, 'saver', 'model.ckpt')
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    self.saver = tf.train.Saver()

    if os.path.isfile(self.saver_path + '.meta'):
      print("Loading variables... ", end='', flush=True)
      self.saver.restore(self.sess, self.saver_path)
    else:
      print("Initializing variables... ", end='', flush=True)
      self.sess.run(self.graph_nodes['init_op'])
    print('complete')

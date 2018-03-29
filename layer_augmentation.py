import tensorflow as tf
import sonnet as snt
import numpy as np

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

input_x = tf.placeholder(tf.float32, shape=[None, 3])
layer_1 = snt.Linear(10, name='layer_1')

layer_2 = snt.Linear(5, name='layer_2')

outputs_1 = layer_1(input_x)
outputs_2 = layer_2(outputs_1)

sess.run(tf.global_variables_initializer())

print(sess.run(outputs_2, {input_x: [[1, 2, 3]]}))

def extend_weights(layer, num_new_outputs):
  weights, biases = layer.get_variables()
  weights = sess.run(weights)
  biases = sess.run(biases)
  total_outputs = biases.shape[0] + num_new_outputs
  new_weights = np.random.normal(size=(weights.shape[0], num_new_outputs))
  new_biases = np.random.normal(loc=0.5, size=num_new_outputs)
  new_biases = np.concatenate((biases, new_biases))
  new_weights = np.concatenate((weights, new_weights), 1)
  inits = {
      'w': tf.constant_initializer(new_weights),
      'b': tf.constant_initializer(new_biases)
  }
  print(total_outputs)
  the_layer = snt.Linear(total_outputs, initializers=inits, name='layer_2b')
  return the_layer

layer_2b = extend_weights(layer_2, 3)
outputs_2b = layer_2b(outputs_1)
sess.run(tf.variables_initializer(layer_2b.get_variables()))

print(sess.run(outputs_2b, {input_x: [[1, 2, 3]]}))

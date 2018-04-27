import tensorflow as tf
import sonnet as snt
import numpy as np
import models.layers as Layers

bin_base = os.path.join('bin', 'source_models', '{}_{}'.format(dataset, source_num_way))

directories = [(os.path.join(bin_base, sub_dir), sub_dir) for sub_dir in os.listdir(bin_base) \
                  if os.path.isdir(os.path.join(bin_base, sub_dir))]

# List of (directory, split) pairs
subdir_splits = []
with open(os.path.join(bin_base, 'idx_splits.csv'), 'r') as csv_file:
  reader = csv.reader(csv_file)
  for row in reader:
    subdir, *split = row
    subdir_splits.append((subdir, [int(s) for s in split]))

print("About to train {} times... that sounds crazy.".format(len(subdir_splits)))


subdir, split = subdir_split
bin_dir = os.path.join(bin_base, subdir)










class Embedder:
  def __init__(self):
    # A list of embedding layers for 2d tensors for reusing when the inputs and embeddings are the same size
    # Key is ((in_height, in_width), (emb_height, emb_width))
    self._embedders_2d = {
    }
    self._placeholders = []
    # Convolution padding doesn't work great if we're not using powers of 2 for the embedding sizes
    self._emb_height = 8
    self._emb_width = 8
    self._kernel_height = 3
    self._kernel_width =3

  def build_placeholders(self, grads_weights):
    '''
    Given a list of pairs of (gradient, weight) tensors, builds and returns a list placeholders of the same shape
    '''
    for g, w in grads_weights:
      name = w.name[:w.name.index('/')]
      gradient_ph = tf.placeholder(tf.float32, g.shape,   name=name + '_grad_placeholder')
      weight_ph = tf.placeholder(tf.float32, w.shape, name=name + '_placeholder')
      self._placeholders.append((gradient_ph, weight_ph))
    total = 0
    for grad, weight in self._placeholders:
      total += np.prod(grad.shape.as_list())
      total += np.prod(weight.shape.as_list())
    print("Number of parameters:", total)
    return self._placeholders

  def _embed_2d(self, in_tensor, emb_height=6, emb_width=6):
    '''
    Given a 2d tensor, embeds it into a fixed-size 2d tensor
    '''
    in_height, in_width = in_tensor.shape.as_list()
    lookup_val = ((in_height, in_width), (emb_height, emb_width))
    if lookup_val not in self._embedders_2d:
      self._embedders_2d[lookup_val] = {
          'horizontal': snt.Linear(emb_width),
          'vertical': snt.Linear(emb_height)
      }
    embedders = self._embedders_2d[lookup_val]
    horizontal_embedder = embedders['horizontal']
    vertical_embedder = embedders['vertical']

    compress_horizontal = lambda x: tf.reshape(horizontal_embedder(tf.expand_dims(x, 0)), [emb_width])
    horizontally_compressed = tf.map_fn(compress_horizontal, in_tensor)

    compress_vertical = lambda x: tf.reshape(vertical_embedder(tf.expand_dims(x, 0)), [emb_height])
    vertically_compressed = tf.map_fn(compress_vertical, tf.transpose(horizontally_compressed))

    return tf.transpose(vertically_compressed)

  def _embed_conv2d(self, in_tensor, emb_height, emb_width):
    '''
    Embeds a conv2d layer's weights, retaining the spatial size and compressing the rest (assuming 3x3 spatial size)
    '''

    kernel_height, kernel_width, in_channels, out_channels = in_tensor.shape.as_list()
    # print(in_tensor.shape)
    in_tensor = tf.reshape(in_tensor, [kernel_height * kernel_width, in_channels, out_channels])

    embed_fn = lambda x: self._embed_2d(x, emb_height, emb_width)
    embedded_tensor = tf.map_fn(embed_fn, in_tensor)
    embedded_tensor = tf.reshape(embedded_tensor, [kernel_height, kernel_width, emb_height, emb_width])
    # print(embedded_tensor.shape)
    return embedded_tensor

  def _embed_conv2d_grads_and_weights(self, grads, weights):
    embedded_grads = self._embed_conv2d(grads, emb_height=self._emb_height, emb_width=self._emb_width)
    embedded_grads_flat = tf.reshape(embedded_grads, [1, -1, 1])
    embedded_weights = self._embed_conv2d(weights, emb_height=self._emb_height, emb_width=self._emb_width)
    embedded_weights_flat = tf.reshape(embedded_weights, [1, -1, 1])
    return tf.concat([embedded_grads_flat, embedded_weights_flat], 2)

  def _unembed_conv2d_weights(self, weights, shape):
    weights = tf.reshape(weights, [self._kernel_height, self._kernel_width, self._emb_height, self._emb_width])
    unembedded_weights = self._embed_conv2d(weights, emb_height=shape[-2], emb_width=shape[-1])
    return unembedded_weights

  def unembed_all_weights(self, weights_list):
    # Separate into a list of tensors (one per gradient/weight pair)
    weights_list = tf.split(weights_list, weights_list.shape.as_list()[0])
    unembedded_grads_weights = []

    for weights, shape in zip(weights_list, self._original_shapes):
      # Split further into individual gradients/weights
      unembedded_grads_weights.append(self._unembed_conv2d_weights(weights, shape))
    return unembedded_grads_weights


  def embed_all_grads_weights(self, grads_weights_list):
    embedded_grads_weights = []
    self._original_shapes = []
    for grads, weights in grads_weights_list:
      self._original_shapes.append(weights.shape.as_list())
      if 'conv' in weights.name:
        embedded_grads_weights.append(self._embed_conv2d_grads_and_weights(grads, weights))
      else:
        raise ValueError('Layer type not supported - convolutional only at this point')
    return tf.concat(embedded_grads_weights, 0)

class Encoder:
  def encode(self, in_tensor):
    in_tensor = snt.Conv2D(output_channels=8, kernel_shape=3, stride=[1, 2], name='encode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2D(output_channels=16, kernel_shape=3, stride=[1, 2], name='encode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2D(output_channels=16, kernel_shape=3, stride=[1, 2], name='encode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2D(output_channels=16, kernel_shape=3, stride=[1, 2], name='encode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2D(output_channels=32, kernel_shape=3, stride=[1, 2], name='encode_conv', padding='SAME')(in_tensor)
    print("Encoded size:", np.prod(in_tensor.shape))
    return in_tensor

  def decode(self, in_tensor):
    in_tensor = snt.Conv2DTranspose(output_channels=32, kernel_shape=3, stride=[1, 2], name='decode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2DTranspose(output_channels=16, kernel_shape=3, stride=[1, 2], name='decode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2DTranspose(output_channels=16, kernel_shape=3, stride=[1, 2], name='decode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2DTranspose(output_channels=16, kernel_shape=3, stride=[1, 2], name='decode_conv', padding='SAME')(in_tensor)
    in_tensor = tf.nn.relu(in_tensor)
    in_tensor = snt.Conv2DTranspose(output_channels=num_classes+additional_classes, kernel_shape=3, stride=[1, 2], name='decode_conv', padding='SAME')(in_tensor)
    in_tensor = snt.Conv2D(output_channels=1, kernel_shape=1, name='decode_conv', padding='SAME')(in_tensor)
    return in_tensor



#### This is where the magic happens ####
embedder = Embedder()
placeholders = embedder.build_placeholders(grads_weights)
embedded_grads_weights = embedder.embed_all_grads_weights(placeholders)
# Fake batching
embedded_grads_weights = tf.expand_dims(embedded_grads_weights, 0)
encoder = Encoder()
encoded = encoder.encode(embedded_grads_weights)
decoded = encoder.decode(encoded)
# Fake batching
decoded = tf.squeeze(decoded, [0])
weight_updates = embedder.unembed_all_weights(decoded)
#### This is where the magic happens ####

print((placeholders[2][1]).shape)
1/0.
outputs = tf.nn.conv2d(inputs, placeholders[0][1] + weight_updates[0], [1, 1, 1, 1], padding='SAME', name='new_conv1')
outputs = tf.nn.relu()
outputs = tf.nn.conv2d(outputs, placeholders[1][1] + weight_updates[1], [1, 1, 1, 1], padding='SAME', name='new_conv2')
outputs = tf.nn.relu()
outputs = tf.nn.conv2d(outputs, placeholders[2][1] + weight_updates[2], [1, 1, 1, 1], padding='SAME', name='new_conv3')
outputs = Layers.global_pool(outputs)
outputs = tf.Print(outputs, [tf.shape(outputs)], "outputs shape", summarize=1000)
predictions = tf.reshape(outputs, [-1, 10])


targets = tf.placeholder(tf.float32, [None, num_classes + additional_classes])
loss = tf.losses.softmax_cross_entropy(targets, predictions)
opt = tf.train.AdamOptimizer(1e-3)
train_op = opt.minimize(loss)

sess.run(tf.global_variables_initializer())
def build_feed_dict(imgs, lbls):
  feed_dict = {}
  for grad, weight in placeholders:
    feed_dict[grad] = np.random.normal(size=grad.shape)
    feed_dict[weight] = np.random.normal(size=weight.shape)
  feed_dict[inputs] = imgs
  feed_dict[targets] = lbls
  return feed_dict
for i in range(2):
  imgs, lbls = next_batch(num_way=num_classes+additional_classes, batch_size=1024)
  l, _ = sess.run([loss, train_op], build_feed_dict(imgs, lbls))
  if i % 100 == 0:
    print(l)





#
#
#
# losses = []
# for (grad, weight), (p_grad, p_weight) in zip(placeholders, unembedded_grads_weights):
#   losses.append(tf.reduce_mean(tf.pow(tf.subtract(grad, p_grad), 2)))
#   losses.append(tf.reduce_mean(tf.pow(tf.subtract(weight, p_weight), 2)))
# loss = tf.reduce_mean(losses)
# opt = tf.train.AdamOptimizer(1e-3)
# train_op = opt.minimize(loss)
# feed_dict = {}
# for grad, weight in placeholders:
#   feed_dict[grad] = np.random.normal(size=grad.shape)
#   feed_dict[weight] = np.random.normal(size=weight.shape)
#
# sess.run(tf.global_variables_initializer())
# for _ in range(10000):
#   l, _ = sess.run([loss, train_op], feed_dict)
#   print(l)
# # for grads_placeholder, weights_placeholder in placeholders:
# #   print("_____")
# #   print(grads_placeholder.shape)
# #   # print(weights_placeholder.shape)
# #   embedded_grads_weights = embed_conv2d_grads_and_weights(grads_placeholder, weights_placeholder)
# #   print(embedded_grads_weights.shape)
# #   # print(tf.reshape(embedded_weights, [-1]).shape)
# #
# # # print(embed_2d(tf.ones([6, 9])))
# # # print(embed_2d(placeholders[0]))
# #
# #
# #


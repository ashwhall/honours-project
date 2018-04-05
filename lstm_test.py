import sonnet as snt
import tensorflow as tf
import numpy as np


tf.reset_default_graph()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)



d_hid = 2 # size of RNN hidden state
outSize = 1
batch_size = 1
d_in = 1
seqLength = 9

input_sequence = tf.placeholder(tf.float32, shape=[seqLength, batch_size, d_in], name='ins')
target = tf.placeholder(tf.float32, shape=[seqLength, batch_size, d_in], name='outs')
lstmCell = snt.LSTM(d_hid, name="lstm")
deepRNNCell = snt.DeepRNN([lstmCell], name="deep_lstm", skip_connections=False)

# the initial state function parameter specifies the batch size for the data.
initialState = deepRNNCell.initial_state(batch_size)
output, final_state =  tf.nn.dynamic_rnn(cell=deepRNNCell, inputs=input_sequence, time_major=True, initial_state=initialState)
output_module = snt.Linear(outSize, name="linear_output")
batch_module = snt.BatchApply(output_module)
final_output = batch_module(output)

loss = tf.reduce_mean(tf.losses.mean_squared_error(final_output, target))
optimize = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(loss)
sess.run(tf.global_variables_initializer())

# print("here is first input with length 3")
# here is the input of a sequence length of 3 with batch size of 1 and 2D input vectors
data = np.asarray([[[0]], [[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]]], np.float32)
# print(sess.run(output, feed_dict={input_sequence : data, target: data}))

# print("\nhere is first input with length 4")
# here is the input of a sequence length of 4 with batch size of 1 and 2D input vectors
data = np.asarray([[[0]], [[1]], [[2]], [[3]], [[4]], [[5]], [[6]], [[7]], [[8]]], np.float32)
# print(sess.run(final_output, feed_dict={input_sequence : data, target: data}))




steps = 1000
for i in range(steps):
    if i % 100 == 0:
        print("{:.2f}%\r".format(100 * i / steps), end='', flush=True)
    _, outs = sess.run([optimize, final_output], feed_dict={input_sequence : data, target: data})
print()
print(outs)

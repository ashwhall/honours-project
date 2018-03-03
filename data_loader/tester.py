import sys
import os
import time
sys.path.append(os.path.abspath('..'))
from data_partitioner import DataPartitioner
from constants import Constants

VID_PATH = "/swimming/2017 SAL Nationals 4K Footage/"
VID_EXTENSIONS = ["mp4", "mov", "avi"]

Constants.config = {
  'video_path': VID_PATH,
  'video_extensions': VID_EXTENSIONS,
  'partition_policy': {
    'frame_percent': 0.75
  },
  'train_queue_size': 20,
  'test_queue_size': 10,
  'event_type': 'backstroke',
  'skip_frames': 2,
  'max_consecutive_frames': 50,
  'n_frames': 5
}

dp = DataPartitioner()

train_interface = dp.get_training_data_interface()
test_interface = dp.get_testing_data_interface()

start = time.time()
count = 0
while True:
  print(train_interface.get_next_batch()['frame_number'])
  count += 1
  if count % 50 == 0:
    print("============================================")
    print((float(count) + count*Constants.config['skip_frames'][0]) / (time.time() - start))
  start = time.time()
  count = 0

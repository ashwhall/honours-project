import csv
import functools
import os
import pickle
import queue
import random
import sys
import threading
import numpy as np
import tensorflow as tf

from base_runner import BaseRunner
from cached_outputs import CachedOutputs
from constants import Constants
from helper import Helper

from data_loader.data_partitioner import DataPartitioner
from data_loader.video_ import Video
from data_loader.video_ import VideoTooShortException

class Forward(BaseRunner):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.graph_nodes = self.graph_builder.build_graph(only_test=True)
    self.label_class = Helper.import_label_class(Constants.config['label_class'])

  def _all_frames(self, video):
    '''
    Generator for batches of frames from the video
    '''
    b = Constants.config['batch_size']
    frames = []
    frame_nums = []
    frame, frame_num = video.get_next_frames(cache=True)
    while frame is not None:
      frames.clear()
      frame_nums.clear()
      for x in range(b):
        if (frame is None):
          return
        frames.append(frame)
        frame_nums.append(frame_num)
        frame, frame_num = video.get_next_frames(cache=True)
      frames_np = np.array(frames)
      frame_nums_np = np.array(frame_nums)
      yield frames_np, frame_nums_np

  def _augment_frames(self, all_frames, all_targets, lbl):
    '''
    Generator that takes batches of frames, gets their associated targets and passes them through
    augmentation
    '''
    augment_options = Constants.config['test_data_augmentation']
    for (frames, frame_nums) in all_frames:
      get_targets = lambda x: lbl.targets(x)
      these_targets = np.array(list(map(get_targets, frame_nums)))
      frames, _ = lbl.augment_data(frames, these_targets, augment_options)
      yield frames, frame_nums

  def _frames_through_model(self, lbl, frames_gen):
    '''
    Pushes all frames through the model; assumes `frames_gen` provides batches
    '''
    all_outputs = []
    all_frame_nums = []
    all_targets = []
    all_losses = []
    for (frames, frame_nums) in frames_gen:
      targets = list(map(lambda x: lbl.targets(x), frame_nums))
      all_targets.extend(targets)
      if 'sparse_targets' in Constants.config and Constants.config['sparse_targets'] == True:
        targets = Helper.densify_sparse_targets(targets)
      output, losses = self.sess.run((
          self.graph_nodes['outputs'],
          self.graph_nodes['loss']
        ), {
          self.graph_nodes['input_x']: frames,
          self.graph_nodes['input_y']: targets,
          self.graph_nodes['is_training']: False
        })
      output = self.graph_builder.get_module().separate_outputs(output)
      all_outputs.extend(output)
      all_frame_nums.extend(frame_nums)
      for _ in frame_nums:
        all_losses.append(float(losses))
      print('Frame #{:7d} processed'.format(frame_nums[0]), end='\r', flush=True)
    print('')
    return all_outputs, all_frame_nums, all_targets, all_losses

  def _get_all_predictions_skip_frame(self, video_filename, label_filename, skip_frames):
    '''
    Pushes frames through the model to generate the predictions and losses
    '''
    n_frames = Constants.config['n_frames']
    downsample = Constants.config['downsample_factor']

    # Create the label and video objects
    lbl = self.label_class(Constants.config['video_path'], video_filename, label_filename, n_frames, skip_frames)
    video_filepath = os.path.join(Constants.config['video_path'], video_filename)
    vid = Video(video_filepath, lbl, n_frames, skip_frames, downsample)

    # Gather frames
    vid.set_start_frame(vid.first_frame)
    all_frames = BackgroundGenerator(self._all_frames(vid))
    # all_frames is a generator, that produces one batch of frames per call

    if ('test_data_augmentation' in Constants.config):
      all_frames = self._augment_frames(all_frames, lbl)

    # frame_nums is collected to make it easier to align outputs to actual frame numbers
    # All outputs are parallel lists of
    outputs, frame_nums, targets, losses = self._frames_through_model(lbl, all_frames)

    return outputs, frame_nums, targets, losses, lbl.inference_targets(), vid.window_size

  def _get_all_predictions(self, video_filename, label_filename):
    '''
    Gets all predictions, frame_nums, targets, losses, inference_targets, window_sizes
    for all the different skip_frame settings.
    '''
    predictions, frame_nums, targets, losses, inference_targets, window_sizes = {}, {}, {}, {}, None, {}
    # The target intervals should be the same, regardless of
    # skip frame setting, so just use the latest one
    for skip_frames in Constants.config['skip_frames']:
      try:
        p, f, t, l, inference_targets, w = self._get_all_predictions_skip_frame(video_filename, label_filename, skip_frames)
        predictions[skip_frames] = p
        frame_nums[skip_frames] = f
        targets[skip_frames] = t
        losses[skip_frames] = l
        window_sizes[skip_frames] = w
      except VideoTooShortException as e:
        print(e)
    return predictions, frame_nums, targets, losses, inference_targets, window_sizes


  def run(self):
    self._start_tf_session()

    data_partitioner = DataPartitioner()

    for (video_filename, label_filename) in data_partitioner.get_test_filenames():
      print('forward passing on: {}'.format(video_filename))

      predictions, frame_nums, targets, losses, inference_targets, window_size = \
        self._get_all_predictions(video_filename, label_filename)

      results = {'predictions': predictions,
        'frame_nums': frame_nums,
        'targets': targets,
        'losses': losses,
        'inference_targets': inference_targets,
        'window_size': window_size}

      CachedOutputs.save(video_filename, results)

class BackgroundGenerator(threading.Thread):
  '''
  A class that wraps a normal generator to start a thread
  and pre-fetch responses asynchronously.
  '''
  def __init__(self, generator):
    threading.Thread.__init__(self)
    self.queue = queue.Queue(20)
    self.generator = generator
    self.daemon = True
    self.start()

  def run(self):
    for item in self.generator:
        self.queue.put(item)
    self.queue.put(None)

  def __iter__(self):
    return self

  def __next__(self):
    next_item = self.queue.get()
    if next_item is None:
      raise StopIteration
    return next_item


def main(argv):
  np.set_printoptions(precision=4, linewidth=150, suppress=True)

  if '--config_file' in argv:
    config_file = argv[argv.index('--config_file') + 1]
  else:
    config_file = 'config_crop15.yml'

  if '--description' in argv:
    description = argv[argv.index('--description') + 1]
  else:
    description = ''

  random.seed(7218)
  tf.set_random_seed(6459)
  np.random.seed(7518)

  forward = Forward(config_file, description, argv)
  forward.run()


if __name__ == "__main__":
  main(sys.argv)


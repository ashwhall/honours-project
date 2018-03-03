import csv
import importlib
import re
import os

from constants import Constants
from helper import Helper
from .video_ import Video
from .video_ import VideoTooShortException
from .jpg_video import JpgVideo

class DataPartitioner:
  '''
  A class that reads two csvs which contain the file names for the training and
  testing sets respectively.
  '''
  TRAIN_FILENAME = 'train.csv'
  TEST_FILENAME = 'test.csv'
  def __init__(self):
    self.root_dir = os.path.join('partitions', Constants.config['dataset'].lower())

    self.label_class = Helper.import_label_class(Constants.config['label_class'])
    self.global_sampler_class = Helper.import_global_sampler_class(Constants.config['global_sampler']['name'])

    self.videos = []
    self.train_filepath = os.path.join(self.root_dir, DataPartitioner.TRAIN_FILENAME)
    self.test_filepath = os.path.join(self.root_dir, DataPartitioner.TEST_FILENAME)

  def get_training_video_sampler(self):
    video_list = []
    for filepath in self.train_filepaths:
      video_list.extend(self._load_file(self.root_dir, filepath))
    return self.global_sampler_class(video_list, **Constants.config['global_sampler']['params'])

  def get_testing_video_sampler(self):
    video_list = []
    for filepath in self.test_filepaths:
      video_list.extend(self._load_file(self.root_dir, filepath))
    return self.global_sampler_class(video_list, **Constants.config['global_sampler']['params'])

  def get_train_filenames(self):
    filename_list = []
    for filename in self.train_filepaths:
      filename_list.extend(self.get_filenames(filename))
    return filename_list

  def get_test_filenames(self):
    filename_list = []
    for filename in self.test_filepaths:
      filename_list.extend(self.get_filenames(filename))
    return filename_list

  def get_filenames(self, filepath):
    '''
    A generator for the filenames in the file at `filepath`.
    '''
    with open(filepath, 'r') as f:
      reader = csv.reader(f, skipinitialspace=True)

      for (video_filename, label_filename) in reader:
        yield (video_filename, label_filename)

  def _load_file(self, root_dir, filename):
    '''
    The file at `filename` should list a set of other filenames.
    Returns a list of sampler objects, one for each line in the file at `filename`.
    '''
    file_pattern = re.compile('\\%[0-9]+d')
    n_frames = Constants.config['n_frames']
    downsample = Constants.config['downsample_factor']
    result = []
    for (i, (video_filename, label_filename)) in enumerate(self.get_filenames(filename)):
      video_filepath = os.path.join(root_dir, video_filename)
      for skip_frames in Constants.config['skip_frames']:
        lbl = self.label_class(root_dir, video_filename, label_filename, n_frames, skip_frames)
        try:
          if (file_pattern.search(video_filepath)):
            vid = JpgVideo(video_filepath, lbl, n_frames, skip_frames, downsample)
          else:
            vid = Video(video_filepath, lbl, n_frames, skip_frames, downsample)
          sampler = self.video_sampler_class(vid, lbl)
          result.append(sampler)
        except VideoTooShortException as e:
          print('{} : {}'.format(video_filepath, e))
      print('loaded {:4d} labels from {}'.format(i+1, filename), end='\r', flush=True)
    print()
    return result

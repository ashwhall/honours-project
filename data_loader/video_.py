import collections
import cv2
import random
import numpy as np

from constants import Constants

class Video: # pylint: disable=R0902
  '''
  This class represents a video file on disk and associated JSON annotation file
  '''
  def __init__(self, video_filepath, label_obj=None, n_frames=1, skip_frames=0, downsample_factor=1):
    self.video_filepath = video_filepath

    if label_obj is not None:
      self.label_obj = label_obj

      # If label knows the frames, get values from it, else tell the
      # label the frame information
      if hasattr(label_obj, 'last_frame'):
        # The label object defines which frames we will read from
        # to account for the case that only part of the video is labelled
        self.first_frame = label_obj.first_frame
        self.last_frame = label_obj.last_frame
      else:
        self.read_frame_info()
        self.label_obj.set_frame_info(self.first_frame, self.last_frame)
    else:
      self.read_frame_info()

    self.frame_count = self.last_frame - self.first_frame + 1

    self.n_frames = n_frames
    self.skip_frames = skip_frames
    self.downsample_factor = downsample_factor
    self.window_size = self._calculate_window_size()

    if (self.frame_count < self.window_size):
      raise VideoTooShortException('Video initialised with n_frames={} and {} skip '
        'frames requires {} frames, but only has {} frames'\
        .format(n_frames, skip_frames, self.window_size, self.frame_count))

    self.curr_frame = 0
    self.start_frame = None
    self.looped = False

    self.cap = None
    self.frame_cache = collections.deque(maxlen=self.n_frames)

  def read_frame_info(self):
    self.first_frame = 0
    cap = cv2.VideoCapture(self.video_filepath)
    self.last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    cap.release()

  def _calculate_window_size(self):
    return (self.n_frames-1)*(1+self.skip_frames) + 1

  def valid_starting_frames(self):
    return (self.first_frame, self.last_frame - self.window_size + 1)


  def _reset_file(self):
    '''
    Reset the video properties for our looping procedure to default
    '''
    self.close_file()
    self.clear_cache()
    self.start_frame = None
    self.curr_frame = None
    self.looped = False

  def _set_curr_frame(self, frame_num):
    '''
    Sets the internal tracker of current frame to a particular frame
    '''
    self.curr_frame = frame_num
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.curr_frame)
    self.clear_cache()

  def _open_file(self):
    '''
    Open the video file so it's ready to go.
    If the current frame is stored, then open at that frame
    '''
    if self.cap is None:
      self.cap = cv2.VideoCapture(self.video_filepath)
      if self.curr_frame is not None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.curr_frame)

  def clear_cache(self):
    if self.frame_cache:
      self.frame_cache.clear()

  def close_file(self):
    '''
    Deallocate video resources
    '''
    if self.cap is not None:
      self.cap.release()
      self.cap = None

  def read(self):
    '''
    Makes a call to the internal VideoCapture object
    '''
    return self.cap.read()

  def do_skip_frame(self):
    '''
    Skips a frame with the internal VideoCapture
    '''
    self.cap.grab()
    self.curr_frame += 1

  def get_next_frame(self):
    '''
    Returns the next frame, the index of that frame, a boolean indicating
    whether there are more frames to come.
    Reading starts at a random frame, and wraps back around to the start as
    required.
    Possible return values:
      frame, curr_frame-1, True  -> More frames to come
      None,            -1, True  -> Video needs to continue from the start
      None,            -1, False -> All frames have been returned (will start from a
                            random frame when called again)
    '''
    # Open the file if it's been closed
    self._open_file()

    # Choose a starting frame if we haven't done so yet
    if self.start_frame is None:
      self.set_start_frame()

    # We've done a full `lap` through the frames
    if self.curr_frame == self.start_frame and self.looped:
      self._reset_file()
      return None, -1, False

    # We're about to read over the end of the video
    if self.curr_frame > self.last_frame:
      self._set_curr_frame(self.first_frame)
      self.looped = True
      return None, -1, True

    # Read and return a frame
    frame_num = self.curr_frame
    status, frame = self.read()
    self.curr_frame += 1

    if (self.downsample_factor != 1):
      frame = frame[::self.downsample_factor, ::self.downsample_factor]

    return frame, frame_num, True

  def get_next_frames(self, cache=False):
    '''
    Returns the next n frames and the index of the first of the n frames
    Returns (None, None) if there are not enough frames left in the video to create
    a collection of frames.
    '''
    frame_num = None

    while len(self.frame_cache) < self.frame_cache.maxlen:
      frame, frame_num, more = self.get_next_frame()

      if frame is not None:
        for _ in range(self.skip_frames):
          self.do_skip_frame()

        self.frame_cache.append(frame)
      else:
        self.clear_cache()

      if not more:
        return None, None


    result = list(self.frame_cache)
    if (cache):
      self.frame_cache.popleft()
    else:
      self.clear_cache()
    idx_of_first = frame_num - self.window_size + 1
    return result, idx_of_first

class VideoTooShortException(Exception):
  pass

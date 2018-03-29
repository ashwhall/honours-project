import collections
import random
import numpy as np
import scipy as sp

from constants import Constants


class DataAugmentation:
  '''
  Collection of stateless data augmentation methods.
  Acts on a whole batch at once.
  '''
  # |frames| = [batch, frames, height, width, channels]
  @staticmethod
  def random_crop(frames, targets, crop_size):
    if type(frames) is np.ndarray:
      x_pos = random.randint(0, frames.shape[3]-crop_size)
      y_pos = random.randint(0, frames.shape[2]-crop_size)
      new_frames = frames[:, :, y_pos:y_pos+crop_size, x_pos:x_pos+crop_size, :]
    elif isinstance(frames, collections.Iterable) and isinstance(frames[0], collections.Iterable):
      x_pos = random.randint(0, frames[0][0].shape[1]-crop_size)
      y_pos = random.randint(0, frames[0][0].shape[0]-crop_size)
      new_frames = []
      for batch_element in frames:
        new_frames.append([frame[y_pos:y_pos+crop_size, x_pos:x_pos+crop_size] for frame in batch_element])
    else:
      raise Exception('frames and frames[0] must be iterable')
    return new_frames, targets

  @staticmethod
  def centre_crop(frames, targets, h, w):
    if type(frames) is np.ndarray:
      s = frames.shape
      h_lower = s[2]//2-h//2
      h_higher = s[2]//2+h//2
      w_lower = s[3]//2-w//2
      w_higher = s[3]//2+w//2
      new_frames = frames[:, :, h_lower:h_higher, w_lower:w_higher]
    elif isinstance(frames, collections.Iterable) and isinstance(frames[0], collections.Iterable):
      new_frames = []
      for batch_element in frames:
        window = []
        for frame in batch_element:
          s = frame.shape
          h_lower = s[0]//2-h//2
          h_higher = s[0]//2+h//2
          w_lower = s[1]//2-w//2
          w_higher = s[1]//2+w//2
          window.append(frame[h_lower:h_higher, w_lower:w_higher])
        new_frames.append(window)
    else:
      raise Exception('frames and frames[0] must be iterable')
    return new_frames, targets

  @staticmethod
  def five_crops(frames, targets, crop_size):
    h = frames.shape[2]
    w = frames.shape[3]
    tl = frames[:, :,    :crop_size,    :crop_size, :]
    bl = frames[:, :, h-crop_size:h,    :crop_size, :]
    tr = frames[:, :,    :crop_size, w-crop_size:w, :]
    br = frames[:, :, h-crop_size:h, w-crop_size:w, :]
    c = frames[:, :,  h//2-crop_size//2:h//2+crop_size//2, w//2-crop_size//2:w//2+crop_size//2, :]
    return np.stack([tl, bl, tr, br, c], 1), targets

  @staticmethod
  def random_jitter(frames, targets, magnitude):
    if type(frames) is np.ndarray:
      new_frames = frames + np.random.uniform(-magnitude, magnitude, frames.shape)
    elif isinstance(frames, collections.Iterable) and isinstance(frames[0], collections.Iterable):
      new_frames = []
      for batch_element in frames:
        new_frames.append([frame + np.random.uniform(-magnitude, magnitude, frame.shape) for frame in batch_element])
    else:
      raise Exception('frames and frames[0] must be iterable')
    return new_frames, targets

  @staticmethod
  def random_zoom(frames, targets, max_zoom):
    if type(frames) is np.ndarray:
      zoom = np.random.uniform(-max_zoom, max_zoom, [frames.shape[0]]+np.ones_like(frames.shape[1:]))
      new_frames = sp.ndimage.interpolation.zoom(frames, 1 + zoom)
    elif isinstance(frames, collections.Iterable) and isinstance(frames[0], collections.Iterable):
      new_frames = []
      for batch_element in frames:
        zoom = np.random.uniform(-max_zoom, max_zoom, [])
        new_frames.append([sp.ndimage.interpolation.zoom(frame, zoom) for frame in batch_element])
    else:
      raise Exception('frames and frames[0] must be iterable')
    return new_frames, targets

  @staticmethod
  def random_lr_flip(frames, targets, prob, flip_targets=False):
    '''
    Performs a left-right flip of the target images with probability `prob`.
    Targets are flipped left-right assuming that their shape is [-1, 3] (as is for swimming labels.
    Thankfully, this processes batches of shape [4, 5, 1080, 1920, 3] in 1e-5 seconds
    '''
    for batch_item_num, batch_item in enumerate(frames):
      flip = np.random.uniform() < prob
      if flip:
        for frame_num, frame in enumerate(batch_item):
          batch_item[frame_num] = frame[:, ::-1, :]
        if flip_targets:
          # Find indices for this batch's targets
          indices = np.where((targets == batch_item_num)[:, 0])
          # Flip the x coordinates
          targets[indices, 2] = Constants.VIDEO_WIDTH - targets[indices, 2]
    return frames, targets

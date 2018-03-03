import contextlib
import os
import pickle
import sys
import numpy as np

from constants import Constants
from cached_outputs import CachedOutputs
from data_loader.labels.swimming_label import SwimmingLabel
from data_loader.data_partitioner import DataPartitioner
from space_transform.space_transform import SpaceTransform


# No stdout from: https://stackoverflow.com/questions/2828953/
class DummyFile(object):
    def write(self, x): pass
@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

class SwimmingEvaluator:

  def __init__(self, config_file):
    Constants.load_config(config_file)

  def lane_metrics_from_filename(self, video_filename, label_filename):

    lbl = SwimmingLabel(Constants.config['video_path'], video_filename, label_filename)

    outputs = CachedOutputs.load(video_filename)
    skip_frames = Constants.config['skip_frames'][0]
    p = outputs['predictions'][skip_frames]
    t = outputs['targets'][skip_frames]
    f = outputs['frame_nums'][skip_frames]

    return self.lane_metrics(p, t, f, lbl)

  def lane_metrics(self, p, t, f, lbl):
    '''
    Lane metrics are recall and mean distance

    Recall is calculated as a measure of how many lanes received a prediction

    Mean distance is the average (across all lanes) distance between the
    predicted point in the lane and the ground truth in that lane.

    In order to aggregte across the whole video we need to decompose recall into
    n_correct and n_truths, and add them up separately, only calculating the
    recall at the end (otherwise smaller videos will be worth the same as longer videos,
    and thus the points in a longer video will individually worth less)

    BIG NOTE: Very occasionally two of the labelled targets will end up in
    the same lane, meaning that the number of targets used to calculate the recall
    will be less than the number of targets provided. This means that n_correct
    will be a non-integer that is less than the true value. This is not handled
    here at all; n_correct will sometimes return a float value that is a
    conservative estimate from ignoring these cases.
    '''
    n_truths, n_correct = 0, 0
    dists = []
    for (preds, targs, frame_nums) in zip(p, t, f):
      if len(targs) == 0:
        continue
      n_truths += len(targs)
      # Need to make batch counts and targets the shape expected by SwimmingLabel:
      #  preds[2] (AKA batch_counts) shape == [1, 2]
      #     (implying one batch, batch #0 with the stored number of predictions)
      #  targs shape = [None, 3], where the 3 is {batch number, y, x}
      preds[2] = [[0, preds[2]]]
      targs = np.array(targs)
      targs = np.pad(targs, ((0,0), (1, 0)), 'constant', constant_values=0)
      with nostdout():
        stats = lbl.evaluate(preds, targs, [frame_nums])
      # Recall = n_correct/n_truths
      #         (i.e. out of the ground truths, what proportion were found)
      if stats['recall'] is not None:
        this_n_correct = len(targs)*stats['recall']
        n_correct += this_n_correct
        # Without knowledge of how close each swimmer is, the best we can do
        #  is assume the same mean_dist for each swimmer (average for frame)
        if stats['mean_distance'] is not None:
          dists.extend([stats['mean_distance']] * int(this_n_correct))

    return n_truths, n_correct, dists


  def get_distances_from_filename(self, video_filename):

    outputs = CachedOutputs.load(video_filename)
    skip_frames = Constants.config['skip_frames'][0]
    p = outputs['predictions'][skip_frames]
    # preds[3] is the np.array of predictions in original image space.
    # We upsample to compare against the raw targets
    p = [preds[3] * Constants.config['downsample_factor'] for preds in p]
    t = outputs['targets'][skip_frames]
    t = [np.array(targ) * Constants.config['downsample_factor'] for targ in t]

    base_dir = Constants.config['video_path']
    st = SpaceTransform(os.path.join(base_dir, video_filename))

    return self.get_distances(p, t, st)

  @staticmethod
  def get_distances(p, t, st):
    '''
    Point-wise distances between predicted coordinates and target coordinates
    (measured in real world metres)

    Args:
        p: model predictions; shaped [frames, ?, 2]; measured in original image space
        t: targets; shaped [frames, ?, 2]; measured in original image space
        st (SpaceTransform): Object for translating to real world measurements

    Returns:
        p_to_t: A list of distances, one per model prediction for which there was
          a target on the same frame.
        t_to_p: A list of distances, one per target for which there was a model
          prediction on the same frame.
        p_with_none (int): count of predictions for which there were no targets
        t_with_none (int): count of targets for which there were no predictions
    '''
    p_to_t, t_to_p = [], []
    p_with_none, t_with_none = 0, 0
    targs_c = []

    for (preds, targs) in zip(p, t):
      if len(targs) == 0:
        p_with_none += len(preds)
        continue
      if len(preds) == 0:
        t_with_none += len(targs)
        continue

      # Model produces (y, x) coords but SpaceTransform expects (x, y) coords
      # So the order is swapped when calling the space transform function
      pred_norm_coords = map(lambda x: st.norm_coords(x[1], x[0]), preds)
      targ_norm_coords = list(map(lambda x: st.norm_coords(x[1], x[0]), targs))
      pred_world_coords = list(map(lambda x: st.real_coords(x[0], x[1]), pred_norm_coords))
      targ_world_coords = list(map(lambda x: st.real_coords(x[0], x[1]), targ_norm_coords))

      # Numpy versions shape: [?, 2]
      pred_world_coords = np.array(pred_world_coords)
      targ_world_coords = np.array(targ_world_coords)

      # Broadcast to find distances between every predicted coord with every target coord
      pred = np.expand_dims(pred_world_coords, 0) # [1, ?, 2]
      targ = np.expand_dims(targ_world_coords, 1) # [?, 1, 2]
      dists = np.linalg.norm(targ-pred, axis=2) # [?, ?] ; euclidean distance == Frobenius norm

      pred_closest = np.min(dists, axis=0)
      targ_closest = np.min(dists, axis=1)

      p_to_t.extend(pred_closest)
      t_to_p.extend(targ_closest)

    return p_to_t, t_to_p, p_with_none, t_with_none

  def evaluate_all(self):
    # Metrics:
    # Precision/Recall at 0.05m/0.10m/0.15m/0.20m
    # Median distance
    dist_thresholds = [0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    # p stands for prediction, t stands for target
    # these dists between points of each kind and the nearest of the other
    dists_p_to_t = []
    dists_t_to_p = []

    total_truths, total_correct = 0, 0
    all_lane_dists = []

    data_partitioner = DataPartitioner()

    for (i, (video_filename, label_filename)) in enumerate(data_partitioner.get_test_filenames()):
      print('evaluating: {:100s}'.format(video_filename), end='\r', flush=True)

      n_truths, n_correct, dists = self.lane_metrics_from_filename(video_filename, label_filename)
      total_truths += n_truths
      total_correct += n_correct
      all_lane_dists.extend(dists)

      p_to_t, t_to_p, p_with_none, t_with_none = self.get_distances_from_filename(video_filename)
      dists_p_to_t.extend(p_to_t)
      dists_t_to_p.extend(t_to_p)

    # Print lane-based metrics
    print('\nOverall lane metrics:')
    print('\t         Recall:   {:8.6f}'.format(total_correct/total_truths))
    print('\t  Mean Distance:   {:8.6f}'.format(np.mean(all_lane_dists)))
    for x in np.linspace(0, 100, 20, endpoint=False):
      print('\t   {:02d}% Distance:   {:8.6f}'.format(int(x), np.percentile(all_lane_dists, x)))

    # Print point-wise-based metrics
    dists_p_to_t.sort()
    dists_t_to_p.sort()
    n_p = len(dists_p_to_t) + p_with_none
    n_t = len(dists_t_to_p) + t_with_none

    mean_dist_p_to_t = np.mean(dists_p_to_t)
    mean_dist_t_to_p = np.mean(dists_t_to_p)

    print('\nOverall point-wise metrics:')
    print('\tp with none / n_p:{:10d}/{:10d}'.format(p_with_none, n_p))
    print('\tt with none / n_t:{:10d}/{:10d}'.format(t_with_none, n_t))
    print('\tmean_dist_p_to_t:     {:8.6f}'.format(mean_dist_p_to_t))
    print('\tmean_dist_t_to_p:     {:8.6f}'.format(mean_dist_t_to_p))
    print('\tp_to_t:')
    for x in np.linspace(0, 100, 20, endpoint=False):
      print('\t\t   {:02d}% Distance:   {:8.6f}'.format(int(x), np.percentile(dists_p_to_t, x)))
    print('\tt_to_p:')
    for x in np.linspace(0, 100, 20, endpoint=False):
      print('\t\t   {:02d}% Distance:   {:8.6f}'.format(int(x), np.percentile(dists_t_to_p, x)))
    for dist_threshold in dist_thresholds:
      p_to_t_mask = np.array(dists_p_to_t) < dist_threshold
      t_to_p_mask = np.array(dists_t_to_p) < dist_threshold
      n_p_to_t_within = np.count_nonzero(p_to_t_mask)
      n_t_to_p_within = np.count_nonzero(t_to_p_mask)

      recall = n_t_to_p_within/n_t
      precision = n_p_to_t_within/n_p

      print('@{:4.2f}m:\t            recall:      {:8.6f}'.format(dist_threshold, recall))
      print('\t_________precision:______{:8.6f}___________'.format(precision))

def main(argv):
  np.set_printoptions(precision=4, linewidth=150, suppress=True)

  if '--config_file' in argv:
    config_file = argv[argv.index('--config_file') + 1]
  else:
    config_file = 'config_crop15.yml'

  evaluator = SwimmingEvaluator(config_file)
  evaluator.evaluate_all()


if __name__ == "__main__":
  main(sys.argv)

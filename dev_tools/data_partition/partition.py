import math
import os
import random
import shutil

def partition_dataset(summary_file, partition_policy, filenames):
  '''
  Splits file_pairs into training_file_pairs and testing_file_pairs based
  on the policy given in the constructor
  '''
  random.shuffle(filenames)
  if 'video_percent' in partition_policy:
    summary_file.write("\nUsing \"video_percent\" partition policy:\n")
    summary_file.write(str(partition_policy))
    summary_file.write('\n')
    split_percent = partition_policy['video_percent']

    # Get a shuffled list of videos
    # Find which index to split at
    split_index = int(math.ceil(split_percent*len(filenames)))
    # Actually split into training and testing
    training_videos = filenames[:split_index]
    testing_videos = filenames[split_index:]
  elif 'frame_percent' in partition_policy:
    summary_file.write('\nUsing \"frame_percent\" partition policy:\n')
    summary_file.write(str(partition_policy))
    summary_file.write('\n')

    split_percent = partition_policy['frame_percent']

    total_frames = sum(lbl_obj.frame_count for (_,_,lbl_obj) in filenames)
    target_frames = int(math.ceil(split_percent*total_frames))

    training_videos = []
    testing_videos = []

    # Add to the training set until we've got our minimum number of frames
    frames_received = 0
    for filepaths in filenames:
      lbl_obj = filepaths[2]
      if frames_received > target_frames:
        testing_videos.append(filepaths)
      else:
        training_videos.append(filepaths)
      frames_received += lbl_obj.frame_count
  else:
    msg = 'Unknown partition policy \"' + partition_policy.keys()[0] + '\"'
    raise ValueError(msg)
  summary_file.write('\nTESTING VIDEOS:\n')
  for (v_file,_,_) in testing_videos:
    summary_file.write(v_file + '\n')
  summary_file.write('\nTRAINING VIDEOS:\n')
  for (v_file,_,_) in training_videos:
    summary_file.write(v_file + '\n')
  summary_file.write('\n')

  for output in [summary_file.write, print]:
    output('Training videos: {:4d}, frames: {:10d}\n'.format(
        len(training_videos),
        sum(lbl_obj.frame_count for (_,_,lbl_obj) in training_videos)))
    output('Testing videos:  {:4d}, frames: {:10d}\n'.format(
        len(testing_videos),
        sum(lbl_obj.frame_count for (_,_,lbl_obj) in testing_videos)))
    output('Total videos:    {:4d}, frames: {:10d}\n'.format(
        len(filenames),
        sum(lbl_obj.frame_count for (_,_,lbl_obj) in filenames)))

  assert len(testing_videos) > 0, \
      'Partition policy resulted in no videos in the test dataset - be more generous!'

  return training_videos, testing_videos


def write_files(folder, partition_policy, filenames, options_str='default'):

  save_folder = os.path.join(folder, '.partition', options_str)
  # Make the folders
  shutil.rmtree(save_folder, ignore_errors=True)
  os.makedirs(save_folder)

  with open(os.path.join(save_folder, 'summary.txt'), 'w') as f:
    training_videos, testing_videos = partition_dataset(f, partition_policy, filenames)

  with open(os.path.join(save_folder, 'train.csv'), 'w') as f:
    for (v_file, l_file, _) in training_videos:
      f.write('\"{}\",\"{}\"\n'.format(v_file, l_file))

  with open(os.path.join(save_folder, 'test.csv'), 'w') as f:
    for (v_file, l_file, _) in testing_videos:
      f.write('\"{}\",\"{}\"\n'.format(v_file, l_file))

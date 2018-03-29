# From the correspondences file, create label instances to find the label for each
# frame and save segment information (as expected by Caffe/C3D)

assert __name__ == "__main__", 'Must be run as a script'

import os
import random
import re
import shutil
import sys

sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from data_loader.labels.thumos_label import ThumosLabel

if not (len(sys.argv) == 2):
  print('usage: python3 thumos_preprocess.py /path/to/data/folder')
  sys.exit(0)

data_folder = sys.argv[1]

N_FRAMES = 16
OVERLAP = 0.75
sample_rates = [1, 2, 4, 8, 16, 32]
correspondence_file = 'correspondence.txt'

class BadProgrammerError(Exception):
  pass

def check_untrimmed(f):
  return bool(f.count('thumos15_video_validation_'))

ucf_pattern = re.compile('jpg/v_[a-zA-Z]{,30}_g[0-9]{2}_c[0-9]{2}/%06d.jpg')
def check_ucf101(f):
  return bool(ucf_pattern.match(f))



# Gather filenames, create Label objects for common interface to reading labels
filenames = []
with open(os.path.join(data_folder, correspondence_file), 'r') as correspondences:
  correspondences = map(lambda x: x.rstrip('\n'), correspondences)
  for (i, (v_f, l_f)) in enumerate(map(lambda x: x.split(','), correspondences)):
    for s in sample_rates:
      lbl_obj = ThumosLabel(data_folder, v_f, l_f, N_FRAMES, s-1)
      filenames.append((v_f, l_f, lbl_obj))
  print('{:5d} labels loaded'.format(i), end='\r', flush=True)
print('')
random.shuffle(filenames)




# Find all positive and negative segments from the labels
positives = []
negatives = []
for (i, (v_f, l_f, lbl_obj)) in enumerate(filenames):
  v_folder = os.path.dirname(os.path.join(data_folder, v_f))
  jpgs = os.listdir(v_folder)
  w_s = lbl_obj.window_size
  segment_frame_nums = list(range(0, len(jpgs)-w_s, int(round(w_s*0.75))))
  prefix = lambda x: os.path.join('feature', os.path.basename(v_folder), '{:06d}'.format(x))
  if check_untrimmed(v_f):
    ious = lbl_obj.iou(segment_frame_nums)
    for (k, (frame_num, iou)) in enumerate(zip(segment_frame_nums, ious)):
      if iou > 0.5:
        positives.append(((v_folder, frame_num+1, 1, lbl_obj.skip_frames+1), prefix(k)))
      else:
        negatives.append(((v_folder, frame_num+1, 0, lbl_obj.skip_frames+1), prefix(k)))
  elif check_ucf101(v_f):
    for (k, frame_num) in enumerate(segment_frame_nums):
      positives.append(((v_folder, frame_num+1, 1, lbl_obj.skip_frames+1), prefix(k)))
  else:
    raise BadProgrammerError('This shouldn\'t happen')
  print('{:7d}/{:7d} files\' segments stored'.format(i, len(filenames)), end='\r', flush=True)
print('')


# Determine shortest subset, and ensure an even number of both
n_pos = len(positives)
n_neg = len(negatives)
if len(positives) < len(negatives):
  random.shuffle(positives)
  negatives = random.sample(negatives, n_pos)
elif len(negatives) > len(positives):
  random.shuffle(negatives)
  positives = random.sample(positives, n_neg)
else:
  random.shuffle(positives)
  random.shuffle(negatives)



# Make the folders
save_folder = os.path.join(data_folder, '.partition', 'plain_list')
shutil.rmtree(save_folder, ignore_errors=True)
os.makedirs(save_folder)


# Actually write to the file
with open(os.path.join(save_folder, 'list.lst'), 'w') as segment_f:
  with open(os.path.join(save_folder, 'prefix.lst'), 'w') as prefix_f:
    n = len(positives)
    for (i, (pos, neg)) in enumerate(zip(positives, negatives)):
      segment_f.write('{} {} {} {}\n'.format(*(pos[0])))
      segment_f.write('{} {} {} {}\n'.format(*(neg[0])))
      prefix_f.write('{}\n'.format(pos[1]))
      prefix_f.write('{}\n'.format(neg[1]))
      print('{:7d}/{:7d} segments written'.format(i, n), end='\r', flush=True)
    print('')

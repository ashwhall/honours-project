import json
import os
import random
import sys

import partition

sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from data_loader.labels.thumos_label import ThumosLabel

correspondence_file = 'correspondence.txt'

N_FRAMES = 16


def _read_files(data_folder):
  '''
  Returns list of (video_filename, lbl_filename, lbl_object)
  '''
  result = []

  with open(os.path.join(data_folder, correspondence_file), 'r') as correspondences:
    correspondences = map(lambda x: x.rstrip('\n'), correspondences)
    for (i, (v_f, l_f)) in enumerate(map(lambda x: x.split(','), correspondences)):
      lbl_obj = ThumosLabel(data_folder, v_f, l_f, N_FRAMES, 0)
      lbl_obj.read_frame_info()
      result.append((v_f, l_f, lbl_obj))
      print('{:6d} videos loaded'.format(i), end='\r', flush=True)
    print()

  return result


if __name__ == "__main__":
  if (not (len(sys.argv) == 3 or len(sys.argv) == 4 or len(sys.argv) == 5)):
    print('usage: python3 thumos_partition.py /path/to/data/folder partition_policy    jpg ')
    print('where:                                   path string           json        False')
    print('example partition_policy: {\\"frame_percent\\": 0.75}')
    sys.exit(0)

  data_folder = sys.argv[1]
  partition_policy = json.loads(sys.argv[2])
  use_jpg = (sys.argv[3] == 'True')
  filenames = _read_files(data_folder)

  if use_jpg:
    partition.write_files(data_folder, partition_policy, filenames, 'jpg_orig')
  else:
    partition.write_files(data_folder, partition_policy, filenames)


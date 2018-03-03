import json
import os
import sys

import partition

sys.path.append(os.path.join(os.getcwd(), '..', '..'))
from data_loader.labels.swimming_label import SwimmingLabel

video_extensions = ["mp4", "mov", "avi", "mxf"]

def _walk_filetree(data_folder, event_type):
  '''
  Walk the file-tree from root_dir and create a list of video/json filename pairs
  '''
  for subdir, _, files in os.walk(data_folder):
    for filename in files:
      filename = os.path.join(subdir, filename)
      extension_index = filename.rindex(".")+1
      if filename[extension_index:].lower() in video_extensions:
        json_filename = filename[:extension_index] + 'json'
        transform_filename = filename[:extension_index - 1] + '_transform.json'
        if os.path.isfile(json_filename) and os.path.isfile(transform_filename):
          relative_filename = filename.lstrip(data_folder)
          relative_json_filename = json_filename.lstrip(data_folder)
          label_obj = SwimmingLabel(data_folder, relative_filename, relative_json_filename)

          # Skip if we've limited the videos to a specific event type
          if not label_obj.is_valid_event_type(event_type):
            continue

          yield relative_filename, relative_json_filename, label_obj


if __name__ == "__main__":
  if (len(sys.argv) != 4):
    print('usage: python3 swimming_partition.py /path/to/data/folder partition_policy event_type')
    print('where:                                   path string           json           string')
    print('example partition_policy: "{\\"frame_percent\\": 0.75}"')
    sys.exit(0)

  data_folder = sys.argv[1]
  partition_policy = json.loads(sys.argv[2])
  event_type = sys.argv[3]
  filenames = list(_walk_filetree(data_folder, event_type))

  partition.write_files(data_folder, partition_policy, filenames, event_type)

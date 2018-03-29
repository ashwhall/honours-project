# Generates video-level annotations, and creates a correspondence text file listing
# video file names and their corresponding label file.

assert __name__ == "__main__", 'Must be run as a script'

import cv2
import numpy as np
import os
import subprocess as sp
import sys

if not (len(sys.argv) == 2):
  print('usage: python3 thumos_preprocess.py /path/to/data/folder')
  sys.exit(0)

data_folder = sys.argv[1]


allowed_classes = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk', 'CliffDiving',
                  'CricketBowling', 'CricketShot', 'Diving', 'FrisbeeCatch', 'GolfSwing',
                  'HammerThrow', 'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                  'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']

def strip_ext(s):
  last_dot = s.rfind('.')
  if (last_dot == -1):
    return s
  else:
    return s[:last_dot]

lbl_folder =           os.path.join(data_folder, 'TH15_Temporal_annotations_validation', 'annotations')
new_lbl_folder =       os.path.join(data_folder, 'generated_annotations')
jpg_dir =              os.path.join(data_folder, 'jpg_orig')
ucf_folder =           os.path.join(data_folder, 'UCF101_orig')
untrimmed_vid_folder = os.path.join(data_folder, 'thumos15_validation_orig')
correspondence_file =  os.path.join(data_folder, 'correspondence.txt')

with open(correspondence_file, 'w') as f:
  pass
if not os.path.exists(new_lbl_folder):
  os.makedirs(new_lbl_folder)
if not os.path.exists(jpg_dir):
  os.makedirs(jpg_dir)


# UCF101
for filename in os.listdir(ucf_folder):
  counts = [filename.count('_{}_'.format(c)) for c in allowed_classes]
  if any(counts):
    c = allowed_classes[int(np.argmax(np.array(counts)))]
    f_name_no_ext = strip_ext(filename)

    # convert to jpg
    filepath = os.path.join(ucf_folder, filename)
    out_folder = os.path.join(jpg_dir, f_name_no_ext)
    file_pattern = os.path.join(out_folder, '%06d.jpg')
    if not os.path.exists(out_folder):
      os.makedirs(out_folder)
      sp.run(['ffmpeg', '-i', filepath, file_pattern])
    n_frames = len(os.listdir(out_folder))

    # write to new label folder
    lbl_filepath = os.path.join(new_lbl_folder, f_name_no_ext + '.txt')
    with open(lbl_filepath, 'w') as f:
      f.write('{} {} {}\n'.format(c, 0, n_frames))
    with open(correspondence_file, 'a') as f:
      f.write('{},{}\n'.format(file_pattern.lstrip(data_folder), lbl_filepath.lstrip(data_folder)))





# Untrimmed video
def read_vid_fps(vid_folder):
  fpses = {}
  filenames = os.listdir(vid_folder)
  for (i,filename) in enumerate(filenames):
    filepath = os.path.join(vid_folder, filename)
    cap = cv2.VideoCapture(filepath)
    fpses[strip_ext(filename)] = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print("{:6d}/{:6d} videos' fps read".format(i, len(filenames)), end='\r', flush=True)
  print()
  return fpses

vid_fps = read_vid_fps(untrimmed_vid_folder)

# Build the labels per video
untrimmed_lbls = {}
for filename in os.listdir(lbl_folder):
  last_ = filename.rfind('_')
  class_name = filename[last_+1:-4]

  filepath = os.path.join(lbl_folder, filename)
  if os.path.isfile(filepath):
    with open(filepath, 'r') as f:
      for line in f:
        (f_name, start_time, end_time) = line.split()
        fps = vid_fps[f_name]
        start_frame = round(fps * float(start_time))
        end_frame = round(fps * float(end_time))
        if (f_name not in untrimmed_lbls):
          untrimmed_lbls[f_name] = []
        untrimmed_lbls[f_name].append((class_name, start_frame, end_frame))


for filename in os.listdir(untrimmed_vid_folder):
  f_name_no_ext = strip_ext(filename)
  if f_name_no_ext not in untrimmed_lbls:
    continue
  filepath = os.path.join(untrimmed_vid_folder, filename)


  # Convert to jpg
  out_folder = os.path.join(jpg_dir, f_name_no_ext)
  file_pattern = os.path.join(out_folder, '%06d.jpg')
  if not os.path.exists(out_folder):
    os.makedirs(out_folder)
    sp.run(['ffmpeg', '-i', filepath, file_pattern])

  # write to new label folder
  lbl_filepath = os.path.join(new_lbl_folder, f_name_no_ext + '.txt')
  with open(lbl_filepath, 'w') as f:
    for row in untrimmed_lbls[f_name_no_ext]:
      f.write('{} {} {}\n'.format(*row))
  with open(correspondence_file, 'a') as f:
    f.write('{},{}\n'.format(file_pattern.lstrip(data_folder), lbl_filepath.lstrip(data_folder)))








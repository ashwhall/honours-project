import os
import cv2
from scipy.ndimage import interpolation as interp
import numpy as np

def _extend_classes(images, labels):
  '''
  Applies rotations to each of the image classes and labels them as distinct classes
  '''
  output_images = np.array(images)
  output_labels = np.array(labels)
  label_count = np.max(labels)

  # Perform all rotations, add new labels
  for idx, angle in enumerate([90, 180, 270]):
    # Create new label values (add a multiple of the count of labels).
    # E.g. [0, 1, 2] -> [0 + 3, 1 + 3, 2 + 3] = [3, 4, 5]
    new_labels = np.array(labels) + (idx + 1) * label_count
    # Copy the old images
    new_images = np.copy(images)
    # Rotate them by `angle`
    for img_idx, img in enumerate(new_images):
      interp.rotate(img, angle, output=new_images[img_idx])
    # Add to the end of the passed-in list
    output_images = np.append(output_images, new_images, 0)
    output_labels = np.append(output_labels, new_labels, 0)
  return output_images, output_labels

def load_datasets(directory):
  '''
  Finds all images and builds their labels
  Labels are designated by the sub-directory in which the images reside.
  Distinct categories exist between train and test.
  '''
  datasets = {}
  for set_name in ['test', 'train']:
    curr_label = 0
    dir_path = os.path.join(directory, set_name)
    paths = list(os.walk(dir_path))
    labels = []
    images = []
    for i, (subdir, _, files) in enumerate(paths):
      print("Reading {} images: {:.2f}%\r".format(set_name, 100 * i / len(paths)), end="", flush=True)
      image_found = False
      for filename in files:
        filename = os.path.join(subdir, filename)
        if '.png' in filename.lower():
          labels.append(curr_label)
          img = cv2.resize(cv2.imread(filename), (28, 28))[:, :, 0]
          img = img.astype('float32').reshape((28, 28, 1)) / 255
          images.append(img)
          image_found = True
      if image_found:
        curr_label += 1
    images = np.asarray(images)
    labels = np.asarray(labels)

    print("Rotating {} images...      \r".format(set_name), end="", flush=True)
    images, labels = _extend_classes(images, labels)

    datasets[set_name] = {
        'images': np.asarray(images),
        'labels': np.asarray(labels)
    }
  return datasets

def print_dataset_info(the_dict):
  print("Data shape:                          ")
  for set_name in the_dict.keys():
    print(set_name)
    for key in the_dict[set_name].keys():
      print(' ', key, ":", the_dict[set_name][key].shape)

# datasets = load_omniglot_datasets('.')
# print_dataset_info(datasets)

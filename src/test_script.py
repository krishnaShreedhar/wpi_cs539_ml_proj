import argparse
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
import numpy as np
from scipy import ndimage
import nibabel as nib


import utils


def _load_model(model_path, **kwargs):
    # TODO: Add load model code
    model = keras.models.load_model(model_path)
    print(model.summary())
    return model

def load_models(list_paths):
    list_models = []
    for index, model_path in enumerate(list_paths):
        model = _load_model(model_path)
        list_models.append(model)

    return list_models

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = np.min(volume)
    max = np.max(volume)
    volume = (volume) / (max)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64 #32
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

# def process_scan(path):
#     """Read and resize volume"""
#     # Read scan
#     volume = read_nifti_file(path)
#     # Normalize
#     volume = normalize(volume)
#     # Resize width, height and depth
#     volume = resize_volume(volume)
#     return volume


mri_type = "FLAIR"
ratings = [0, 1]
project_folder = "project_folder_correct_" + mri_type + "/" #os.getcwd() + '/data/project_folder_FLAIR/'
paths_test = []
y_test = []
for rating in ratings:
    p = project_folder + 'cross_val_folds/test/' + str(rating) + "/"
    files = os.listdir(p)
    for f in files:
      y_test.append(float(rating))
      paths_test.append(p+f)
print("Paths test: ", paths_test)
print(len(paths_test), len(y_test))
print()

# Using utils.process_scan to keep code modular
patient_scans_test = np.array([utils.process_scan(path) for path in paths_test])

list_paths = ['Semester Project/3d_image_classification.h5']
model_list = load_models(list_paths)
model = model_list[0]
for x in range(len(patient_scans_test)):
    predictions = model.predict(np.expand_dims(patient_scans_test[x],axis=0))
    print()
    print('-----------------')
    print()
    print(paths_test[x], predictions)
    print()
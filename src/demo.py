import argparse
import os
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import pandas as pd
import numpy as np
from scipy import ndimage
import nibabel as nib
from sklearn.metrics import roc_auc_score
import cv2


def _load_model(model_path, **kwargs):
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

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


mri_type = "FLAIR" # what mri type assessing
dset = 'test'# where to find your data to assess
models_dir = 'Semester Project/models/dummy_test/' # pretrained models to do predictions
class_thresh = 0.5 # probability threshold for classification

# get list of models, images to assess, and ground truth labels of images
models_list = os.listdir(models_dir)
list_paths = []
for l in range(len(models_list)):
    list_paths.append(models_dir+str(models_list[l]))

ratings = [0, 1]
project_folder = "project_folder_correct_" + mri_type + "_sample" + "/" #os.getcwd() + '/data/project_folder_FLAIR/'
paths_test = []
y_test = []
for rating in ratings:
    p = project_folder + 'cross_val_folds/' + dset + '/' + str(rating) + "/"
    files = os.listdir(p)
    for f in files:
      y_test.append(float(rating))
      paths_test.append(p+f)

# process the incoming data
patient_scans_test = np.array([process_scan(path) for path in paths_test])

# load the model(s) to use for prediction
model_list = load_models(list_paths)

# predict the class of the incoming image using the pretrained model and show to user a central slice of the
# image and its ground truth (for demo) and predicted methylation status
for m in range(0,1): #model_list:
    model = model_list[m]
    for x in range(0,1): #range(len(patient_scans_test)):
        predictions = model.predict(np.expand_dims(patient_scans_test[x], axis=0))

        image = patient_scans_test[x]
        gt = y_test[x]

        # Assign predicted class labels with provided threshold
        if predictions < class_thresh:
            pred_class = 0.0
        else:
            pred_class = 1.0

        print("Ground Truth: ", gt)
        print("Predicted Label: ", pred_class)
        print()

        gt_txt = "Ground Truth: " + str(gt)
        pred_txt = "Predicted Label: " + str(pred_class)

        imS = cv2.resize(image[:, :, 32], (256, 256))

        coordinates1 = (110, 225)
        coordinates = (110, 245)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        color = (255, 0, 255)
        thickness = 1
        img1 = cv2.putText(imS, gt_txt, coordinates1, font, fontScale, color, thickness, cv2.LINE_AA)
        img = cv2.putText(imS, pred_txt, coordinates, font, fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow("FLAIR", img)
        cv2.waitKey(0)
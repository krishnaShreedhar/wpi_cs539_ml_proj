import os
# import zipfile
import numpy as np
import pandas as pd
import math
import cv2
import shutil
import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers

#import nibabel as nib
#from scipy import ndimage
#import random
#import matplotlib.pyplot as plt

# Define these parameters
# fold_num = 1 # fold_x to process and run model with
folds = 5 # 5 # number of cross val folders to make for train/val split
test_split_percent = 0.1
# train_split_percent = 0.85
# val_split_percent = 1 - train_split_percent
mri_type = "T1w" # the MRI scan type you want to split and use for the model
# project_folder = "C:\\Users\\Nick\\Desktop\\WPI\\Machine Learning\\Semester Project\\"
# patient_folder = project_folder + 'Nifti2\\' # generated nii files from dcm2jpg.py
project_folder = "project_folder_correct_" + mri_type + "/"
if not os.path.isdir(project_folder):
    os.makedirs(project_folder)
patient_folder = 'Nifti/' # generated nii files from dcm2jpg.py
label_csv = "train_labels.csv" #'E:\\rsna-miccai-brain-tumor-radiogenomic-classification\\train_labels.csv'
# NOTE: added column in train_labels.csv to pad patient IDs with zero to agree with folder naming convention

# Training parameters (for current simple model provided by the Jupyter notebook)
epochs = 1 #100
batch_size = 2
w_width = 128
h_height = w_width
d_depth = 64
initial_lr = 0.0001 # initial learning rate
decay_steps = 100000 # # decay steps in the exponential learning rate scheduler
decay_rate = 0.96 # decay rate for the exponential learning rate scheduler

###########################################################
###########################################################

# Load Data -- custom for us, data already downloaded -- preprocess using dcm2jpg.py to get .nii files
mri_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
ratings = ['0', '1'] # possible ratings an image can have
#patients = os.listdir(patient_folder)
df_labels = pd.read_csv(label_csv, dtype = str) # str to keep zero padded patient IDs

labels = df_labels['MGMT_value'].tolist()

#print(patients)
#print(len(patients))
print()
print(labels)
print()
print(df_labels)

# def read_nifti_file(filepath):
#     """Read and load volume"""
#     # Read file
#     scan = nib.load(filepath)
#     # Get raw data
#     scan = scan.get_fdata()
#     return scan
#
# # NOTE: what normalization should we use?????????
# def normalize(volume):
#     """Normalize the volume"""
#     # volume = volume
#     # min = -1000
#     # max = 400
#     # volume[volume < min] = min
#     # volume[volume > max] = max
#     min = np.min(volume)
#     max = np.max(volume)
#     # volume = (volume - min) / (max - min)
#     volume = (volume) / (max) # we could experiment with normalization texhniques
#     # volume = sklearn.preprocessing.normalize(volume)
#     volume = volume.astype("float32")
#     return volume
#
#
# # NOTE: how should we resize??? How many slices to keep...need to find minimum and maximum slices across all scans for each scan type
# def resize_volume(img):
#     """Resize across z-axis"""
#     # Set the desired depth
#     desired_depth = 64
#     desired_width = 128
#     desired_height = 128
#     # Get current depth
#     current_depth = img.shape[-1]
#     current_width = img.shape[0]
#     current_height = img.shape[1]
#     # Compute depth factor
#     depth = current_depth / desired_depth
#     width = current_width / desired_width
#     height = current_height / desired_height
#     depth_factor = 1 / depth
#     width_factor = 1 / width
#     height_factor = 1 / height
#     # Rotate
#     img = ndimage.rotate(img, 90, reshape=False)
#     # Resize across z-axis
#     img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
#     return img
#
#
# def process_scan(path):
#     """Read and resize volume"""
#     # Read scan
#     volume = read_nifti_file(path)
#     # Normalize
#     volume = normalize(volume)
#     # Resize width, height and depth
#     volume = resize_volume(volume)
#     return volume


img_names = []
patients = df_labels["BraTS21ID_padded"].tolist()
for i_name in patients:
    img_names.append(i_name+".nii")
print(img_names)
print()
print()

# labels = ["0", "1", "0", "1", "0", "1", "0", "1", "1"] # temporary dummy labels, COMMENT THIS OUT WHEN RUN FOR REAL

X_trainval, X_test, y_trainval, y_test = train_test_split(img_names, labels, test_size=test_split_percent, random_state=7, shuffle=True, stratify=labels)
print("Train+Val and Test Split")
print("Train+Val Images: ", X_trainval)
print("Test Images: ", X_test)
print()
print("Train+Val Labels:", y_trainval)
print("Test Labels: ", y_test)
print()
print(len(X_trainval))
print(len(X_test))
print()
print("-----------------------")
print()

# Test Set
for testindex in range(0, len(X_test)):

    img_name = X_test[testindex]
    im_label = y_test[testindex]

    folder_name = project_folder + 'cross_val_folds/'
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    test_fold = folder_name + 'test/'
    if not os.path.isdir(test_fold):
        os.makedirs(test_fold)

    for cls in labels:
        class_x = test_fold + cls + "/"
        if not os.path.isdir(class_x):
            os.makedirs(class_x)
    try:
        shutil.copy(patient_folder + img_name[0:5] + "/" + mri_type + "/" + img_name, test_fold + im_label + "/" + str(img_name))
    except Exception:
        print("This image does not exist as Nifti file: ", img_name)

###########
###########
# Train and Val splitting
skf = StratifiedKFold(n_splits = folds, random_state = 7, shuffle = True)
train_index_list = []
valid_index_list = []
for train_index, valid_index in skf.split(np.zeros(len(X_trainval)), y_trainval):
    train_index_list.append(list(train_index))
    valid_index_list.append(list(valid_index))

print()
print("TRAIN:", train_index_list)
print()
print("VALID:", valid_index_list)
# print(len(valid_index_list))
print()

# Make Train folders
folder = 1
for tlist in train_index_list:
    for tindex in tlist:
        img_name = X_trainval[tindex]
        im_label = y_trainval[tindex]

        folder_name = project_folder + 'cross_val_folds/fold_' + str(folder) + "/"
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        train_fold = folder_name + 'train/'
        if not os.path.isdir(train_fold):
            os.makedirs(train_fold)

        for cls in labels:
            class_x = train_fold + cls + "/"
            if not os.path.isdir(class_x):
                os.makedirs(class_x)

        try:
            shutil.copy(patient_folder + img_name[0:5] + "/" + mri_type + "/" + img_name, train_fold + im_label + "/" + str(img_name))
        except Exception:
            print("This image does not exist as Nifti file: ", img_name)

    folder = folder + 1


# Make Val folders
folder = 1
for vlist in valid_index_list:
    for vindex in vlist:
        img_name = X_trainval[vindex]
        im_label = y_trainval[vindex]

        folder_name = project_folder + 'cross_val_folds/fold_' + str(folder) + "/"
        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        val_fold = folder_name + 'val/'
        if not os.path.isdir(val_fold):
            os.makedirs(val_fold)

        for cls in labels:
            class_x = val_fold + cls + "/"
            if not os.path.isdir(class_x):
                os.makedirs(class_x)
        try:
            shutil.copy(patient_folder + img_name[0:5] + "/" + mri_type + "/" + img_name, val_fold + im_label + "/" + str(img_name))
        except Exception:
            print("This image does not exist as Nifti file: ", img_name)

    folder = folder + 1


# #########
# #########
# #########
#
#
# # Processing .nii images for train/val/test sets
# paths_test = []
# for rating in ratings:
#     p = 'cross_val_folds/test/' + rating + "/"
#     file = os.listdir(p)
#     for f in file:
#         # maybe have try-except here to deal with missing .nii files????
#         paths_test.append(p+f)
# print()
# print("Paths test: ", paths_test)
# print(len(paths_test))
# print()
#
# patient_scans_test = np.array([process_scan(path) for path in paths_test])
#
# paths_train = []
# y_train = []
# for rating in ratings:
#     p = 'cross_val_folds/' + 'fold_' + str(fold_num) + "/" + 'train/' + rating + "/" # how to identify which fold_ ???
#     file = os.listdir(p)
#     for f in file:
#         # maybe have try-except here to deal with missing .nii files????
#         y_train.append(float(rating))
#         paths_train.append(p+f)
# print()
# print("Paths train:", paths_train)
# print(len(paths_train))
# print()
#
# patient_scans_train = np.array([process_scan(path) for path in paths_train])
#
# paths_val = []
# y_val = []
# for rating in ratings:
#     p = 'cross_val_folds/' + 'fold_' + str(fold_num) + "/" + 'val/' + rating + "/"
#     file = os.listdir(p)
#     for f in file:
#         # maybe have try-except here to deal with missing .nii files????
#         y_val.append(float(rating))
#         paths_val.append(p+f)
# print()
# print("Paths val:", paths_val)
# print(len(paths_val))
# print()
#
# patient_scans_val = np.array([process_scan(path) for path in paths_val])
#
# x_train = patient_scans_train
# y_train = np.array(y_train)
# x_val = patient_scans_val
# y_val = np.array(y_val)
#
# print()
# print()
# print("------------------------------------------------")
# print("Paths train:", paths_train)
# print("Length x_train: ", len(x_train))
# print("y_train: ", y_train)
# print("Paths val:", paths_val)
# print("Length x_val: ", len(x_val))
# print("y_val: ", y_val)
# print("------------------------------------------------")
# print()
# print()

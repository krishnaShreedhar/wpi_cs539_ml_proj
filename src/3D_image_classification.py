import os
# import zipfile
import numpy as np
import pandas as pd
import math
import cv2
import shutil
import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import nibabel as nib
from scipy import ndimage
import random
import matplotlib.pyplot as plt

# # Download url of normal CT scans.
# url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
# filename = os.path.join(os.getcwd(), "CT-0.zip")
# keras.utils.get_file(filename, url)
#
# # Download url of abnormal CT scans.
# url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
# filename = os.path.join(os.getcwd(), "CT-23.zip")
# keras.utils.get_file(filename, url)
#
# # Make a directory to store the data.
# os.makedirs("MosMedData")
#
# # Unzip data in the newly created directory.
# with zipfile.ZipFile("CT-0.zip", "r") as z_fp:
#     z_fp.extractall("./MosMedData/")
#
# with zipfile.ZipFile("CT-23.zip", "r") as z_fp:
#     z_fp.extractall("./MosMedData/")

# Define these parameters
fold_num = 1  # fold_x to process and run model with
folds = 2  # 5 # number of cross val folders to make for train/val split
test_split_percent = 0.1
# train_split_percent = 0.85
# val_split_percent = 1 - train_split_percent
mri_type = "T2w"  # "T1w" # the MRI scan type you want to split and use for the model
# project_folder = "C:\\Users\\Nick\\Desktop\\WPI\\Machine Learning\\Semester Project\\"
# patient_folder = project_folder + 'Nifti2\\' # generated nii files from dcm2jpg.py
project_folder = "project_folder_" + mri_type + "/"
if not os.path.isdir(project_folder):
    os.makedirs(project_folder)
patient_folder = 'Nifti/'  # generated nii files from dcm2jpg.py
label_csv = "train_labels.csv"  # 'E:\\rsna-miccai-brain-tumor-radiogenomic-classification\\train_labels.csv'
# NOTE: added column in train_labels.csv to pad patient IDs with zero to agree with folder naming convention

# Training parameters (for current simple model provided by the Jupyter notebook)
epochs = 1  # 100
batch_size = 2
w_width = 128
h_height = w_width
d_depth = 64
initial_lr = 0.0001  # initial learning rate
decay_steps = 100000  # # decay steps in the exponential learning rate scheduler
decay_rate = 0.96  # decay rate for the exponential learning rate scheduler

###########################################################
###########################################################

# Load Data -- custom for us, data already downloaded -- preprocess using dcm2jpg.py to get .nii files
mri_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
ratings = ['0', '1']  # possible ratings an image can have
patients = os.listdir(patient_folder)
df_labels = pd.read_csv(label_csv, dtype=str)  # str to keep zero padded patient IDs

labels = df_labels['MGMT_value'].tolist()

print(patients)
print(len(patients))
print()
print(labels)
print()
print(df_labels)


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


# NOTE: what normalization should we use?????????
def normalize(volume):
    """Normalize the volume"""
    # volume = volume
    # min = -1000
    # max = 400
    # volume[volume < min] = min
    # volume[volume > max] = max
    min = np.min(volume)
    max = np.max(volume)
    # volume = (volume - min) / (max - min)
    volume = (volume) / (max)  # we could experiment with normalization texhniques
    # volume = sklearn.preprocessing.normalize(volume)
    volume = volume.astype("float32")
    return volume


# NOTE: how should we resize??? How many slices to keep...need to find minimum and maximum slices across all scans for each scan type
def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
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


# paths = []
# for patient in patients:
#     p = patient_folder + patient + "/" + mri_type + "/"
#     file = os.listdir(p)
#     for f in file:
#         paths.append(p+f)
# print()
# print(paths)
# print(len(paths))
# print()


# # Folder "CT-0" consist of CT scans having normal lung tissue,
# # no CT-signs of viral pneumonia.
# normal_scan_paths = [
#     os.path.join(os.getcwd(), "MosMedData/CT-0", x)
#     for x in os.listdir("MosMedData/CT-0")
# ]
# # Folder "CT-23" consist of CT scans having several ground-glass opacifications,
# # involvement of lung parenchyma.
# abnormal_scan_paths = [
#     os.path.join(os.getcwd(), "MosMedData/CT-23", x)
#     for x in os.listdir("MosMedData/CT-23")
# ]
#
# print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
# print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))


# # Read and process the scans.
# # Each scan is resized across height, width, and depth and rescaled.
# abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths])
# normal_scans = np.array([process_scan(path) for path in normal_scan_paths])


# patient_scans = np.array([process_scan(path) for path in paths])


# # For the CT scans having presence of viral pneumonia
# # assign 1, for the normal ones assign 0.
# abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
# normal_labels = np.array([0 for _ in range(len(normal_scans))])
#


# # Split data in the ratio 70-30 for training and validation.
# x_train = np.concatenate((abnormal_scans[:70], normal_scans[:70]), axis=0)
# y_train = np.concatenate((abnormal_labels[:70], normal_labels[:70]), axis=0)
# x_val = np.concatenate((abnormal_scans[70:], normal_scans[70:]), axis=0)
# y_val = np.concatenate((abnormal_labels[70:], normal_labels[70:]), axis=0)
# print(
#     "Number of samples in train and validation are %d and %d."
#     % (x_train.shape[0], x_val.shape[0])
# )
#
#
#

img_names = []
# patients = df_labels["BraTS21ID_padded"].tolist()
for i_name in patients:
    img_names.append(i_name + ".nii")
print(img_names)
print()
print()

##labels = ["0", "1", "0", "1", "0", "1", "0", "1", "1"] # temporary dummy labels, COMMENT THIS OUT WHEN RUN FOR REAL

X_trainval, X_test, y_trainval, y_test = train_test_split(img_names, labels, test_size=test_split_percent,
                                                          random_state=7, shuffle=True, stratify=labels)
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

    shutil.copy(patient_folder + img_name[0:5] + "/" + mri_type + "/" + img_name,
                test_fold + im_label + "/" + str(img_name))

###########
###########
# Train and Val splitting
skf = StratifiedKFold(n_splits=folds, random_state=7, shuffle=True)
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

        shutil.copy(patient_folder + img_name[0:5] + "/" + mri_type + "/" + img_name,
                    train_fold + im_label + "/" + str(img_name))

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

        shutil.copy(patient_folder + img_name[0:5] + "/" + mri_type + "/" + img_name,
                    val_fold + im_label + "/" + str(img_name))

    folder = folder + 1

# Processing .nii images for train/val/test sets
paths_test = []
for rating in ratings:
    p = project_folder + 'cross_val_folds/test/' + rating + "/"
    file = os.listdir(p)
    for f in file:
        # maybe have try-except here to deal with missing .nii files????
        paths_test.append(p + f)
print()
print("Paths test: ", paths_test)
print(len(paths_test))
print()

patient_scans_test = np.array([process_scan(path) for path in paths_test])

paths_train = []
y_train = []
for rating in ratings:
    p = project_folder + 'cross_val_folds/' + 'fold_' + str(
        fold_num) + "/" + 'train/' + rating + "/"  # how to identify which fold_ ???
    file = os.listdir(p)
    for f in file:
        # maybe have try-except here to deal with missing .nii files????
        y_train.append(float(rating))
        paths_train.append(p + f)
print()
print("Paths train:", paths_train)
print(len(paths_train))
print()

patient_scans_train = np.array([process_scan(path) for path in paths_train])

paths_val = []
y_val = []
for rating in ratings:
    p = project_folder + 'cross_val_folds/' + 'fold_' + str(fold_num) + "/" + 'val/' + rating + "/"
    file = os.listdir(p)
    for f in file:
        # maybe have try-except here to deal with missing .nii files????
        y_val.append(float(rating))
        paths_val.append(p + f)
print()
print("Paths val:", paths_val)
print(len(paths_val))
print()

patient_scans_val = np.array([process_scan(path) for path in paths_val])

x_train = patient_scans_train
y_train = np.array(y_train)
x_val = patient_scans_val
y_val = np.array(y_val)

print()
print()
print("------------------------------------------------")
print("Paths train:", paths_train)
print("Length x_train: ", len(x_train))
print("y_train: ", y_train)
print("Paths val:", paths_val)
print("Length x_val: ", len(x_val))
print("y_val: ", y_val)
print("------------------------------------------------")
print()
print()


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

# Augment them on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(batch_size)
        .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(batch_size)
        .prefetch(2)
)

data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the MRI scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
plot_slices(4, 10, 128, 128, image[:, :, :40])


def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=w_width, height=h_height, depth=d_depth)
model.summary()

# Compile model.
initial_learning_rate = initial_lr
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

print()
print("DONE")

# fig, ax = plt.subplots(1, 2, figsize=(20, 3))
# ax = ax.ravel()
#
# for i, metric in enumerate(["acc", "loss"]):
#     ax[i].plot(model.history.history[metric])
#     ax[i].plot(model.history.history["val_" + metric])
#     ax[i].set_title("Model {}".format(metric))
#     ax[i].set_xlabel("epochs")
#     ax[i].set_ylabel(metric)
#     ax[i].legend(["train", "val"])


# # Load best weights.
# model.load_weights("3d_image_classification.h5")
# prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
# scores = [1 - prediction[0], prediction[0]]
#
# class_names = ["normal", "abnormal"]
# for score, name in zip(scores, class_names):
#     print(
#         "This model is %.2f percent confident that CT scan is %s"
#         % ((100 * score), name)
#     )

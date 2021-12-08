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

# def _load_model(model_path, **kwargs):
#     list_str_summary = []
#     model = keras.models.load_model(model_path)
#     model.summary(print_fn=lambda x: list_str_summary.append(f"{x}"))
#     print('\n'.join(list_str_summary))
#     return model
#
#
# def load_models(list_paths):
#     list_models = []
#     for index, dict_model_path in enumerate(list_paths):
#         model = _load_model(dict_model_path["path"])
#         list_models.append(model)
#
#     return list_models

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
    desired_depth = 32 #64
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


mri_type = "FLAIR"
dset = 'test'
models_dir = 'Semester Project/models/dummy_test/'
# list_paths = ['Semester Project/3d_image_classification.h5']
class_thresh = 0.5 # threshold for deciding between class labels in sigmoid output

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

print("Paths test: ", paths_test)
print(len(paths_test), len(y_test))
print()
print()

patient_scans_test = np.array([process_scan(path) for path in paths_test])

print('Model list: ', models_list)
print()
print(list_paths)

model_list = load_models(list_paths)
# model = model_list[0]
model_type = 0
for model in model_list:
    # model.summary()
    print()
    score = model.evaluate(patient_scans_test, np.array(y_test), batch_size=2)
    print("Loss of %0.4f" % score[0])
    print("Accuracy of %0.4f" % score[1]) # based on 50% threshold, default
    print()
    pred_class_list = []
    pred_probs = []
    images = []
    for x in range(len(patient_scans_test)):
        predictions = model.predict(np.expand_dims(patient_scans_test[x], axis=0))
        pred_probs.append(predictions[0][0])
        images.append(paths_test[x])

        # Assign predicted class labels with provided threshold
        if predictions < class_thresh:
            pred_class = 0.0
        else:
            pred_class = 1.0

        pred_class_list.append(pred_class)
        print('-----------------')
        print(paths_test[x], predictions)
        print()

    print()
    print("Ground Truths: ", y_test)
    print("Class Predictions: ", pred_class_list) # based on provided threshold
    print()

    print("pred probs: ", pred_probs)
    roc_auc = roc_auc_score(y_test, pred_probs)

    print("ROC_AUC: ", roc_auc)

    df = pd.DataFrame(columns=["Images", "Pred Prob", "GT Label", "Pred Label", "Overall ACC", "Overall Loss", "Overall AUC", "Model"])
    df["Images"] = pd.Series(images)
    df["Pred Prob"] = pd.Series(pred_probs)
    df["GT Label"] = pd.Series(y_test)
    df["Pred Label"] = pd.Series(pred_class_list)
    df["Overall ACC"] = pd.Series(score[1])
    df["Overall Loss"] = pd.Series(score[0])
    df["Overall AUC"] = pd.Series(roc_auc)
    df["Model"] = pd.Series(str(models_list[model_type]))

    df.to_csv(mri_type + "_" + str(model_type) + "_thresh_" + str(class_thresh) + ".csv")
    model_type = model_type + 1

import os
# import zipfile
import numpy as np
import pandas as pd
import math
#import cv2
import shutil
import sklearn

mri_type = 'FLAIR'
csv_file_path = "dimensions/" + mri_type + ".csv" #'C:/Users/Nick/Downloads/dimensions/dimensions/' + mri_type + ".csv"
gt_labels_path = "train_labels.csv" #'C:/Users/Nick/Desktop/WPI/Machine Learning/Semester Project/train_labels.csv'


csv_file = pd.read_csv(csv_file_path)
filepaths = csv_file["Filepath"].tolist()
dim1 = csv_file["dim1"].tolist()
dim2 = csv_file["dim2"].tolist()
dim3 = csv_file["dim3"].tolist()

gt_labels = pd.read_csv(gt_labels_path, dtype=str)
id_padded = gt_labels["BraTS21ID_padded"].tolist()
mgmt_labels = gt_labels["MGMT_value"].tolist()

print(id_padded)

base_fold = "new_data/"
if not os.path.isdir(base_fold):
    os.makedirs(base_fold)

mri_folder = base_fold + mri_type + "/"
if not os.path.isdir(mri_folder):
    os.makedirs(mri_folder)

new_data_paths = []
labels = []
for p in range(len(filepaths)):
    fpath = filepaths[p]
    patient_id = str(fpath[6:11])
    # print(patient_id)
    new_data_path = mri_folder + patient_id + ".nii"
    new_data_paths.append(new_data_path)

    shutil.copy(fpath, new_data_path)

    # if patient_id in id_padded:
    idx = id_padded.index(patient_id)
    labels.append(mgmt_labels[idx])

df = pd.DataFrame(columns=["Filepath", "GT Label", "dim1", "dim2", "dim3"])
df["Filepath"] = pd.Series(new_data_paths)
df["GT Label"] = pd.Series(labels)
df["dim1"] = pd.Series(dim1)
df["dim2"] = pd.Series(dim2)
df["dim3"] = pd.Series(dim3)

df.to_csv(base_fold + mri_type + ".csv")


print()
print("labels: ", labels)
print(len(labels))
print(new_data_paths)
print(len(new_data_paths))

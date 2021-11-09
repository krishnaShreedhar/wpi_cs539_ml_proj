# Script for converting all Dicom (dcm) folders for each scan type for each patient to Nifti (nii) files

import dicom2nifti
##import cv2
##import numpy as np
import os
##from nibabel import load as load_nii
import pandas as pd


patient_folder = 'train/' # path to the mri train data
##patient_folder = 'temp/'
patient_list = os.listdir(patient_folder)
print(patient_list)
print(len(patient_list))
print()
print("----------------------")
print()
mri_types = ['FLAIR', 'T1w', 'T1wCE', 'T2w']
df = pd.DataFrame(columns=["Failed Patient", "Failed Scan Type"])
for patient in patient_list:
    print(patient)
    print()
    for mri_type in mri_types:
        print(mri_type)
        original_dicom_directory = patient_folder + patient + '/' + mri_type + '/'
        output_file = 'Nifti/' + patient + '/' + mri_type + '/'
        if not os.path.isdir(output_file):
            os.makedirs(output_file)
        try:
            dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file+patient+".nii", reorient_nifti=True)
        except Exception:
            print("Failed patient: ", patient)
            print("Failed scan type: ", mri_type)
            print()
            log_patient_scan = {'Failed Patient': patient, 'Failed Scan Type': mri_type}
            df_appended = df.append(log_patient_scan, ignore_index=True)
    print()
df_appended.to_csv("log_failed_patient_scan.csv")

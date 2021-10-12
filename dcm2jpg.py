# INSTRUCTIONS TO RUN
# download a patient Tw1 folder from the google drive, unzip file, rename parent folder to Tw1-00000, then run
# pip install all packages below:
     # pip install dicom2nifti cv2 numpy os nibabel
# then run
# observe the "Nifti" and "JPG" folders created in the directory you are in
# "Nifti" is a 3D file forma similar to DICOM except it strips away personal info that could be in the DICOM image
    # DICOM (.dcm) is a file format used on medical image scanners like MRI that stores metadata about patients and patient scans
# "JPG" folder contains each slice of the MRI image converted to a JPG image


import dicom2nifti
import cv2
import numpy as np
import os
from nibabel import load as load_nii


# Pick one MRI image type to use, define the type as a variable
# loop through each patient and grab the defined MRI scan type folder you want for that patient

## something like the beloow can be used to loop through all patients and convert all at once
# mri_type = 'FLAIR'
# for patient in patient_list: # patient list from csv file
#     original_dicom_directory = patient + '/' + mri_type + '/'
#     output_file = 'Nifti/' + patient + '/' + mri_type + '/'
#     jpg_output_folder = 'JPG' + patient + '/' + mri_type + '/'


# temporary for 1 patient
original_dicom_directory = 'T1w-00000/T1w/' # renamed folder structure for patient 00000 (just for demo here)
output_file = 'Nifti/T1w/'
jpg_output_folder = 'JPG/Tw1/'

if not os.path.isdir(output_file):
    os.makedirs(output_file)

if not os.path.isdir(jpg_output_folder):
    os.makedirs(jpg_output_folder)

dicom2nifti.dicom_series_to_nifti(original_dicom_directory, output_file, reorient_nifti=True)

nifti_img = os.listdir(output_file)
img = load_nii(output_file + nifti_img[0])
print(img.shape)
print()
print()

img_data = img.get_fdata()
# header = img.header
# nb_img = header.get_data_shape()
# nb_img_h = nb_img[2] #Hauteur

# print(img_data)
# print()
# print(header)
# print()
# print(nb_img)
# print()
# print(nb_img_h)

for i in range(len(img_data[:,:,3])):

    imgSlice = img_data[:,:,i]

    if np.max(imgSlice) != 0:
        imgSlice = (imgSlice / np.max(imgSlice)*255).astype(np.uint8)
    else:
        imgSlice = imgSlice.astype(np.uint8)

    cv2.imwrite(jpg_output_folder + str(i+1) + '.jpg', imgSlice)

    # cv2.imshow('slice1', imgSlice)
    # cv2.waitKey()

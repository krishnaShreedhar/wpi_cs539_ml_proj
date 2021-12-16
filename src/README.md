# Program Execution instructions


dcm2nii.py -- 
Run this file first.
This file loads the dicom (dcm) images downloaded from Kaggle and converts all patients files for all mri types to nifti (nii) files used during training.
Data will be found in a folder called Nifti/
NOTE: this takes some time to run and the data is large

data_splitting.py -- 
Run this file second. This will will split the nifit data from above into train/val/test sets for each mri type
NOTE: this also takes a bit of time to run and will take up a large amount of memeory. We recommend using a storage source that can support large volumes of data.

3d_model_training.py -- 
This file is used to run load, preprocess, augment and train the simple 3D CNN and ResNet models. Model (.h5) files are saved for best epoch.
See run_train_param.sh for sample submission scripts to run this file with possible input parameters

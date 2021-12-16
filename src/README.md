# Program Execution instructions

pip3 install -r requirements.txt

See to make sure the proper CUDA/CuDNN versions are loaded for training with GPU

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


test_script.py -- 
Use this file to load trained models (.h5 files) to assess model performance on your test set data. Output the results (metrics and predicted labels) to a csv file.

demo.py -- 
Demo code that loads sample data you want to assess and predict a label for. It loads a trained model of your specification, loads and processes the data, and outputs an image with the ground truth (for demo purposes), and a predicted label for that image
Change lines 78-81 to fit your data/models
Define what mri type you want to display and predict
Define where the data is located you want to predict and assess
Define the path to the models you want to use to do the predictions on your data
Define a classification threshold to use for assigning "0" and "1" classes, default 0.5

flair_demo.PNG -- 
A sample output image from demo.py

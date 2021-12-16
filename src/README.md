# Program Execution instructions

pip3 install -r [../requirements.txt](../requirements.txt)

See [installation_instr.md](installation_instr.md) to make sure the proper CUDA/CuDNN and Python versions are loaded for training with GPU

[dcm2nii.py](dcm2nii.py)
- Run this file first.
- This file loads the dicom (dcm) images downloaded from Kaggle and converts all patients files for all mri types to nifti (nii) files used during training.
- Data will be found in a folder called Nifti/
- NOTE: This takes some time to run and the data is large

[data_splitting.py](data_splitting.py)
Run this file second. This will split the nifti data from above into train/val/test sets for each mri type
NOTE: 
- This also takes a bit of time to run and will take up a large amount of memory. 
- We recommend using a storage source that can support large volumes of data.

[3d_model_training.py](3d_model_training.py)
- This file is used to run load, preprocess, augment and train the simple 3D CNN and ResNet models. Model (.h5) files are saved for best epoch.
- Change line 630 to the appropriate path that your data folders from data_splitting.py is output to. 
- See [run_train_param.sh](run_train_param.sh) for sample submission scripts to run this file with possible input parameters

Example command: 
```shell
python3 3d_model_training.py --model_to_train=3dcnn --epochs=1 --initial_lr=0.01 --decay_steps=100000 --decay_rate=0.96 --patience=50 --verbose=2 --w_width=128 --d_depth=32 --classes=2 --fold_num=1 --mri_type=T1w --max_data=4 --batch_size=2
```


[test_script.py](test_script.py)
- Use this file to load trained models (.h5 files) to assess model performance on your test set data. 
- Output the results (metrics and predicted labels) to a csv file.

[demo.py](demo.py)
- Demo code that loads sample data you want to assess and predict a label for. 
- It loads a trained model of your specification, loads and processes the data, and outputs an image with the ground truth (for demo purposes), and a predicted label for that image
- Change lines 78-81 to fit your data/models
- Define what mri type you want to display and predict
- Define where the data is located you want to predict and assess
- Define the path to the models you want to use to do the predictions on your data
- Define a classification threshold to use for assigning "0" and "1" classes, default 0.5

[flair_demo.PNG](flair_demo.PNG)
- A sample output image from demo.py

[std_dimensions.py](std_dimensions.py)
- Script to select eligible scans based on dim1=192, dim2=256, dim3=60.

[new_data_shree.py](new_data_shree.py)
- After selecting standardized data of a certain dimension threshold size, combine the filepath to each of those images, thieir ground truth labels, and dimension sizes into one csv file to be used for training.

[data_handler.py](data_handler.py)
- This will make a list of the patient ID (and relevant data) that has all kinds of MRI scans available for the ensemble method.

[train_ensemble_models.py](train_ensemble_models.py)
- This file is used to run to create ensemble data using pretrained models and train the simple sklearn models.
- Uses [ensemble.py](ensemble.py) to create the data and stored in the ../outputs directory.
- See [run_train_sklearn.sh](run_train_sklearn.sh) for sample submission script.

Example command: 
```shell
python3 train_ensemble_models.py
```

[constants.py](constants.py)
- Defines constant directory paths and values that will be used throughout the codebase

[utils.py](utils.py)
- Defines reusable functions 
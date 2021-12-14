# Brain Tumor Radiogenomic Classification

WPI CS 539 ML Final Project

Felipe Mejias, Jose Raul Gamez Carias, Shreedhar Kodate, Khatera Alizada, Nicholas Josselyn

### <ins>Project Goal</ins>: Determine methylation status of brain tumors using magnetic resonance imaging (MRI)

## Introduction

Automated identification of brain tumors in MRI has many profound clinical applications in surgical treatment planning, image-guided interventions, monitoring tumor growth, and the generation of radiotherapy maps for treatment plans. However, manual identification of brain tumors on MRI is time-consuming, tedious, and subjective and at times it can require invasive follow-up surgeries [1]. Radiologists perform a manual, qualitative approach to identify tumors on MRI, which becomes impractical when dealing with a large number of patients. The automated solution will expedite the process and will decrease the cost of the procedure. The patients, who would otherwise have to wait for longer periods of time because of manual identification procedures by radiologists, could be diagnosed in a timely manner and start their treatment sooner, saving precious time that could extend their lives.


## Background

There is an existing work that ensembles multiple architectures trained on the BraTS data set, however, they do so in order to generate high quality segmentations of brain tumors. In our work, we intend to ensemble architectures, across various MRI scan types, for the classification of methylation status of the tumor. The promise of using ensemble techniques is shown as this implementation won the BraTS competition for segmentation in 2017 [2]. Additionally, work has been done using traditional machine learning to extract features and ensemble methods for classification [3]. However, we propose to remove the manual feature extraction step and employ deep learning models to learn the features to extract and ensemble them to build a reliable classifier. There also exists a work that uses a different MRI dataset with deep learning and ensemble methods for assessing MRI image quality. This work tunes a CNN by adapting and adjusting the VGG architecture to fit their data and ensembles models that were trained on one MRI scan type, but across different scan orientations (coronal, sagittal, and axial) [4].


## Methodology

### Description of the dataset

The data consists of the MRI scans of 585 patients from the BraTS 2021 dataset that labels the MGMT status of the patients. The BraTS 2021 dataset is a comprehensive collection of brain tumor MRI scans of patients from various institutions that use different equipment and imaging protocols that represent heterogeneous image quality and diverse clinical practices from different institutions [1,5]. The data is essentially balanced with 47% being labeled ‘0’ (unmethylated) and 53% being labeled ‘1’ (methylated). As part of the BraTS 2021 data set, the participants are provided with binary class labels denoting their tumor methylation status. Each patient has 4 different scan types (T1w, T2w, FLAIR, and T1wCE) each of which captures unique information of the tumors.

![image](https://user-images.githubusercontent.com/12739451/145852597-dcdd18a9-e724-496a-89d6-ae2ff4687be3.png)

### 3D FLAIR Brain MRI

![3D FLAIR Brain MRI](https://user-images.githubusercontent.com/12739451/146030278-55b10a5c-526e-4610-8e2e-86556095f4bb.gif)

### 3D Convolutional Neural Network

3D Convolutional Neural Networks have been found to be useful for image classification problems. Conv3D is mostly used with 3D image data such as Magnetic Resonance Imaging (MRI) data. These artificial neural networks are made up of convolutional, pooling and fully-connected layers. 3D convolutional neural networks are characterised by three main properties: local connectivity of the hidden units, parameter sharing and the use of pooling operations.

![image](https://user-images.githubusercontent.com/12739451/145865045-44dea87f-2e6f-4025-a160-974fd3b404d9.png)

### ResNet50

3D-ResNet50 model is a powerful deep neural networks which has achieved amazing performance results in image classification. The architecture of ResNet50 has 4 stages with a sequence of (3, 4, 6, 3) residual blocks with 3 layers each as shown in the diagram. The addition of the identity connection does not introduce extra parameters. Therefore, the computation complexity for simple deep networks and deep residual networks is almost the same.

![image](https://user-images.githubusercontent.com/12739451/145860742-63448cc4-4f92-46a6-9780-6f0e0bba5007.png)

### Ensemble Approach

The ensemble approach used build high quality classifiers individually, then concatenate the learned features for each best MRI scan type model, and fine tune an ensemble classifier across 4 MRI scan types.

![image](https://user-images.githubusercontent.com/12739451/145855540-71ff0eda-8d6d-4294-9aec-5f3755e45186.png)

### Machine Learning Approach

![image](https://user-images.githubusercontent.com/12739451/145855429-dd4e8275-8f78-4452-8650-cb5a6d8a29f4.png)

## Proposed Experiments

- 4 MRI types
  - Simple 3D-CNN, ResNet-50
- 4-fold cross-validation
  - Bullet list item 2 
- Hyperparameter tuning
  - Batch Size
  - Learning Rate
  - Learning Rate Scheduler (exponential decay)
  - Optimizer (SGD, Adam)
  - Number of filters
  - Input image size
- Data augmentation
  - Random rotation
- Ensemble model
  - Best model for each MRI type
  - Variety of classifiers

## Results

### Results - Individual Scans

These were the results using a Simple 3D-CNN:

MRI Scan Type | AVG Val Accuracy | AVG Val Loss | AVG Val AUC | AVG Test Accuracy | AVG Test Loss | AVG Test AUC
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
T1w  | 0.54 ± 0.04 | 0.69 ± 0.002 | 0.54 ± 0.03 | 0.52 ± 0.03 | 0.70 ±.0.01 | 0.51 ± 0.02
T1wCE  | 0.54 ± 0.03 | 0.69 ± 0.01 | 0.57 ± 0.06 | 0.52 ± 0.01 | 0.69 ± 0.001 | 0.49 ± 0.11
T2w  | 0.53 ± 0.01 | 0.69 ± 0.001 | 0.52 ± 0.04 | 0.54 ± 0.01 | 0.69 ± 0.001 | 0.50 ± 0.04
FLAIR  | 0.54 ± 0.04 | 0.69 ± 0.004 | 0.56 ± 0.05 | 0.55 ± 0.05 | 0.69 ± 0.003 | 0.51 ± 0.02


These were the results using ResNet-50:

MRI Scan Type | AVG Val Accuracy | AVG Val Loss | AVG Val AUC | AVG Test Accuracy | AVG Test Loss | AVG Test AUC
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
T1w  | 0.53 ± 0.02 | 0.69 ± 0.003 | 0.54 ± 0.06 | 0.55 ± 0.03 | 0.69 ± 0.01 | 0.49 ± 0.12
T1wCE  | 0.53 ± 0.005 | 0.69 ± 0.005 | 0.56 ± 0.02 |0.53 ± 0.0 | 0.69 ± 0.002 | 0.51 ±0.04
T2w  | 0.53 ± 0.03 | 0.69 ± 0.002 | 0.62 ± 0.02 | 0.52 ± 0.02 | 0.69 ± 0.001 | 0.49 ± 0.04
FLAIR  | 0.54 ± 0.04 | 0.68 ± 0.02 | 0.59 ± 0.06 | 0.54 ± 0.04 | 0.69 ± 0.004 | 0.55 ± 0.04


### Results - Ensemble Approach

Taking the best performing 4 models on the validation set we ensembled them using the final 64 unit dense layer output to train a downstream classifier. After ensembling the 4 models as feature extractors, and finetuning on a training set and evaluating on an independent test set, a Naive Bayes classifier performs best along with k nearest neighbors and decision tree being more balanced in its predictions.

|   |   |
|---|---|
![image](https://user-images.githubusercontent.com/12739451/145863527-502b97a0-f1fc-4057-a7cd-8c273d394a7d.png)  |  ![image](https://user-images.githubusercontent.com/12739451/145863537-2f191ead-b94b-44cf-8f57-312dd27f5787.png)

### Results - Ensemble - Standardized

|   |   |
|---|---|
![image](https://user-images.githubusercontent.com/12739451/145863495-773e3400-2fbe-406c-bf3c-54c577b1a892.png)  |  ![image](https://user-images.githubusercontent.com/12739451/145863515-7e56b973-9315-4bbe-a125-f9488f504971.png)

## Conclusion

- Best model on validation set:
  - T1w – Simple CNN
  - T1wCE – Simple CNN
  - T2w – ResNet-50
  - FLAIR – ResNet-50
- Best performing MRI type on test set: FLAIR
- Ensemble outperforms in accuracy by 1%
- More standardization/preprocessing of data needed
- Additional data augmentation – limited data

## References

[1] Baid, U. et al., “The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification”, arXiv:2107.02314 (2021)

[2] Kamnitsas, K. et al., “Ensembles of Multiple Models and Architectures for Robust Brain Tumor Segmentation”, International MICCAI brain lesion workshop (2017)

[3] Gupta, N. et al., “Glioma detection on brain MRIs using texture and morphological features with ensemble learning”, Biomedical Signal Processing and Control (2019)

[4] Sujit, S. et al., “Automated image quality evaluation of structural brain MRI using an ensemble of deep learning networks”, Journal of Magnetic Resonance Imaging (2019)

[5] RSNA-MICCAI Brain Tumor Radiogenomic Classification - Kaggle 

[6] https://github.com/hasibzunair/3D-image-classification-tutorial 

# Diabetic-Retinopathy-Grading-Group-34
Diabetic Retinopathy Grading using Deep Learning (PyTorch): A machine learning project that classifies retinal fundus images into diabetic retinopathy severity levels using CNN and Vision Transformer models trained on the Kaggle DR dataset.
# Diabetic Retinopathy Grading using ConvNeXt-Tiny

This branch contains the implementation of a **deep learning pipeline for diabetic retinopathy grading** using **ConvNeXt-Tiny** in PyTorch. The project is based on the **Kaggle Diabetic Retinopathy Detection** dataset and focuses on multi-class classification of retinal fundus images into five severity levels.

## Project Overview

Diabetic retinopathy is a diabetes-related eye disease that can lead to vision loss if not detected early. This project builds an automated image classification model to predict the severity of diabetic retinopathy from retinal fundus photographs.

The model used in this branch is **ConvNeXt-Tiny**, fine-tuned for a 5-class classification task.

## Classes

The dataset labels correspond to the following diabetic retinopathy stages:

- **0** - No DR
- **1** - Mild
- **2** - Moderate
- **3** - Severe
- **4** - Proliferative DR

## Model Used

- **Backbone:** ConvNeXt-Tiny
- **Framework:** PyTorch
- **Pretrained Weights:** ImageNet pretrained ConvNeXt-Tiny
- **Final Layer:** Modified to output 5 classes

## Features of This Branch

- Pretrained **ConvNeXt-Tiny** model
- GPU-enabled training using **CUDA**
- Mixed precision training using **AMP**
- Stratified train-validation split
- Data augmentation for training images
- Weighted sampling to reduce class imbalance impact
- Validation using:
  - Loss
  - Accuracy
  - Weighted F1-score
- Best model checkpoint saving
- Visualization support for:
  - Training and validation loss
  - Training and validation accuracy
  - Training and validation F1-score
  - Confusion matrix
  - Per-class precision, recall, and F1-score
  - Sample predictions and misclassified images

## Project Structure

```bash
├── AML Data Preprocessing.ipynb             # Main notebook for preprocessing, training, validation, and analysis
├── README.md                                # Project documentation
├── best_convnext_tiny.pth                   # Saved best model weights
├── diabetic-retinopathy-convnext-tiny.ipynb # Processed training images archive
├── trainLabels.csv                          # Labels file
├── train_dataset.csv                        # Saved train split
├── training_history                         # Model Performance per Epoch
└── val_dataset.csv                          # Saved validation split
```

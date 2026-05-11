# Diabetic-Retinopathy-Grading-Group-34

Diabetic Retinopathy Grading using Deep Learning (PyTorch): A machine learning project that classifies retinal fundus images into diabetic retinopathy severity levels using CNN and Vision Transformer models trained on the Kaggle Diabetic Retinopathy Detection dataset.

# Diabetic Retinopathy Grading using ConvNeXt-Tiny Ordinal Model

This branch contains the implementation of a deep learning pipeline for diabetic retinopathy grading using **ConvNeXt-Tiny** in PyTorch. The project is based on the **Kaggle Diabetic Retinopathy Detection** dataset and focuses on predicting diabetic retinopathy severity from retinal fundus images.

The final version of this branch uses an **ordinal cumulative prediction setup** instead of a standard flat 5-class classifier. This is more suitable for diabetic retinopathy grading because the target classes have a natural severity order.

## Project Overview

Diabetic retinopathy is a diabetes-related eye disease that can lead to vision loss if not detected early. This project builds an automated image classification model to predict the severity level of diabetic retinopathy from retinal fundus photographs.

The model used in this branch is **ConvNeXt-Tiny**, fine-tuned using an ordinal cumulative head. Instead of treating the five classes as unrelated categories, the model learns ordered severity thresholds between diabetic retinopathy grades.

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
- **Pretrained Weights:** ImageNet-pretrained ConvNeXt-Tiny
- **Task Type:** Ordinal classification
- **Output:** 4 cumulative ordinal logits for 5 DR severity grades
- **Loss Function:** BCEWithLogitsLoss-based cumulative ordinal loss
- **Input Size:** 320 × 320
- **Final Prediction:** Decoded using threshold-based ordinal prediction

## Latest Training Configuration

The latest notebook uses the following configuration:

| Setting | Value |
|---|---:|
| Image size | 320 × 320 |
| Number of classes | 5 |
| Ordinal thresholds | 4 |
| Batch size | 4 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 |
| Optimizer | AdamW |
| Backbone learning rate | 2e-5 |
| Head learning rate | 1e-4 |
| Weight decay | 2e-4 |
| Target smoothing | 0.02 |
| Backbone freeze period | First 3 epochs |
| Training history recorded | 16 epochs |
| Split strategy | Train / Validation / Test |
| Threshold tuning | Enabled |
| Mixed precision | Enabled |
| Channels-last memory format | Enabled |

## Dataset Split

The latest notebook uses a train-validation-test split.

| Split | Number of Images |
|---|---:|
| Train | 24,588 |
| Validation | 5,269 |
| Test | 5,269 |
| Total matched images | 35,126 |

## Class Distribution

| Class | Severity | Total Images |
|---:|---|---:|
| 0 | No DR | 25,810 |
| 1 | Mild | 2,443 |
| 2 | Moderate | 5,292 |
| 3 | Severe | 873 |
| 4 | Proliferative DR | 708 |

## Features of This Branch

- Pretrained **ConvNeXt-Tiny** backbone
- Ordinal cumulative prediction head
- GPU-enabled training using CUDA
- Mixed precision training using AMP
- Train-validation-test split
- Data augmentation for training images
- BCEWithLogitsLoss-based ordinal loss
- Target smoothing for ordinal labels
- Differential learning rates for backbone and head
- Backbone freezing during early training
- Gradient accumulation for effective batch size control
- Validation-based threshold tuning
- Model evaluation using:
  - Loss
  - Accuracy
  - Weighted F1-score
  - Quadratic Weighted Kappa (QWK)
- Best model checkpoint saving
- Visualization support for:
  - Training and validation loss
  - Training and validation accuracy
  - Training and validation weighted F1-score
  - Training and validation QWK
  - Confusion matrix
  - Per-class precision, recall, and F1-score
  - Sample predictions and misclassified images

## Final Ordinal Model Results

The final ordinal ConvNeXt-Tiny model achieved the following test-set results:

| Metric | Test Result |
|---|---:|
| Accuracy | 72.97% |
| Weighted F1-score | 0.7379 |
| Quadratic Weighted Kappa | 0.7395 |

QWK is especially important for this task because diabetic retinopathy grades are ordered. A prediction that is one grade away from the true label is less severe than a prediction that is several grades away.

## Project Structure

```bash
├── AML Data Preprocessing.ipynb              # Main notebook for preprocessing, training, validation, testing, and analysis
├── README.md                                 # Project documentation
├── best_convnext_tiny.pth                    # Saved ConvNeXt-Tiny model checkpoint
├── training_history.csv                      # Earlier flat-classification training history
├── training_history_ordinal.csv              # Ordinal model training history
├── ordinal_validation_test_metrics.csv       # Final ordinal validation/test metrics
├── train_dataset.csv                         # Saved training split
├── val_dataset.csv                           # Saved validation split
├── test_dataset.csv                          # Saved test split
└── trainLabels.csv                           # Original label file
```

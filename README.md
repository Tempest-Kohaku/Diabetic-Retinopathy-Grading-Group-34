# Diabetic-Retinopathy-Grading-Group-34
Diabetic Retinopathy Grading using Deep Learning (PyTorch): A machine learning project that classifies retinal fundus images into diabetic retinopathy severity levels using CNN and Vision Transformer models trained on the Kaggle DR dataset.

## My plan using EfficientNetB3:
1. Work on preprocessing the data on Kaggle notebook so it reduces the size from 89GBs to something more manageable on my system - [x]
2. Work on creating stratified train / test / validation splits from the preprocessed images and labels - [x]
3. Work on handling class imbalance using weighted CrossEntropyLoss - [x]
4. Work on creating a custom Dataset class, Data Loaders and Data augmentations for training of model - [x]
5. Work on loading and setting up EfficientNetB3 - [x]
6. Update EfficientNetB3 to EfficientNetB5 - [ ]
7. Redo preprocessing since input size is different between the models - [ ]
8. Load and set up EfficientNetB5 - [ ]
9. Work on training EfficientNetB5 on the preprocessed dataset - [ ]
10. Work on evaluating the trained model on validation and test sets using suitable metrics for diabetic retinopathy grading - [ ]

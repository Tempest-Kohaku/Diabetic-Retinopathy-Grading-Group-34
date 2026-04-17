# Diabetic-Retinopathy-Grading-Group-34
Diabetic Retinopathy Grading using Deep Learning (PyTorch): A machine learning project that classifies retinal fundus images into diabetic retinopathy severity levels using CNN and Vision Transformer models trained on the Kaggle DR dataset.

## My plan using EfficientNetB5:
1. Work on preprocessing the data on Kaggle notebook so it reduces the size from 89GBs to something more manageable on my system                                                                                                      -   [x]
2. Work on creating stratified train / test / validation splits from the preprocessed images and labels     -   [x]
3. Work on handling class imbalance using weighted CrossEntropyLoss                                         -   [x]
4. Work on creating a custom Dataset class, Data Loaders and Data augmentations for training of model       -   [x]
5. Work on loading and setting up EfficientNetB3                                                            -   [x]
6. Update EfficientNetB3 to EfficientNetB5                                                                  -   [x]
7. Redo preprocessing since input size is different between the models                                      -   [x]
8. Load and set up EfficientNetB5                                                                           -   [x]
9. Work on training EfficientNetB5 on the preprocessed dataset                                              -   [x]
10. Visualize training curves (losses, accuracy)                                                            -   [x]
11. Log training and test data                                                                              -   [x]
12. Experiment with adding warmup schedulers, changing optimisers, LRs, weight_decays, batch_sizes, etc     -   [ ] 
    1. Batch Size = 16, LR = 3e-4, Weight_Decay = 1e-4, AdamW Optimizer, Cosine Scheduling                          [x]
    2. Rotate and flip the test images and average softmax result                                                   [x]
    3. Linear Warmup before Cosine using LR=5e-4 using sequential scheduler [ ]
    4. Use WeightedRandomSampler with unweighted Cross Entropy for sample-based criterion                             [ ]
    5. Change criterion to treat problem as a single class problem by using MSELoss or SmoothL1Loss criterion. Have a similar regression head instead of CrossEntropyLoss.  
    6. Add label-smoothing to CrossEntropyLoss                                                                      [ ]
    7.  Try accumulating steps to avoid OOM after increasing batch size to 64ish and learning rate to 6e-4ish   [ ]
13. Visualise evaluation                                                                                    -   [x]

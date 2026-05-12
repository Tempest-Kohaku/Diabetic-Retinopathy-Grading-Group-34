# Diabetic-Retinopathy-Grading-Group-34
Diabetic Retinopathy Grading using Deep Learning (PyTorch): A machine learning project that classifies retinal fundus images into diabetic retinopathy severity levels using CNN and Vision Transformer models trained on the Kaggle DR dataset.

Simanta 

DeiT(Data-Efficient Image Transformer):

I am using two model:  the Data-Efficient Image Transformer(DeiT-small) model WITH & WITHOUT distillation which has 22.1M parameters.Dataset from the Diabetic Retinopathy Detection competition on Kaggle has been preprocessed in two distinct ways to reduce the size of the dataset for the experiments:
        1.Resizing (224 × 224), minimal normalization, extracted the huge compressed archive in chunks and saved them as JPEG at quality 75.  After that, split logic is class-wise and image-based where each DR class is shuffled and divided into 60% train, 20% validation, and 20% test.(only used in Exp -1)
        2.Data preprocessing with extracts the label CSV to create a patient-level split(considering left right eye ) with preserved class distribution, image resizing to 384 × 384 , saves them at quality 85, center cropping to remove non-retinal borders, ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), and structured train(60%)/validation(20%)/test(20%) splits :
                              Train: 21,074 images, 10,537 patients 
                              Validation: 7,026 images, 3,513 patients 
                              Test: 7,026 images, 3,513 patients
  Training Configuration:
      •	Model: DeiT-Small (22M parameters, 12 transformer blocks, 8 heads) [deit_small_patch16_224]
      •	Teacher: DeiT-Base with ImageNet-21K pre-training (RegNetY-160)
      •	Model 2: DeiT Small Distilled [deit_small_distilled_patch16_224]:
      •	Optimizer: AdamW with weight decay
      •	Loss: Cross-entropy plus KL-divergence distillation
      •	Batch Size: 32 / 16 
      •	Temperature for knowledge distillation
      •	scheduler = CosineAnnealingLR
      •	mixed precision enabled
      •	early stopping driven by validation QWK
      •	final evaluation on the held-out test set
      •	distilled test-time prediction formed by averaging the class head and distillation head logits
      •	Drop Path Rate = 0.1

In total, nine experiments were done by modifying traing and data preprocessing .Quadratic Weighted
Kappa is used as primary metric as it suited for ordinal labels. Different augmentation like random flips, random
rotations, colour jitter and affine transforms, sharpness adjustments and random cropping is applied accordingly.
Validation and test predictions were converted to ordinal grades either by taking the argmax or threshold tuning on the
validation set to maximise QWK.I received the best performing of the models in experiment 6:

Training Data Augmentations:
Resize to 224×224
Random Horizontal Flip (p=0.5)
Random Rotation (20°)
Random Affine Transform (translate: 5%, scale: 95-105%)
Color Jitter (brightness: 0.25, contrast: 0.25, saturation: 0.15)
Random Sharpness Adjustment (p=0.3, factor: 1.5)
ToTensor + Normalize

Validation/Test Data Augmentations:
Resize to 224×224
ToTensor + Normalize (No Random Augmentations)


Experiment 6 results:.	
Acc NoD          0.7798	
Acc Dist         0.7949
BalAcc NoD	     0.4525
BalAcc Dist      0.4805
MacroF1 NoD      0.4808	
MacroF1 Dist     0.5103	
WeightedF1 NoD	 0.7489	
WeightedF1 Dist  0.7600
QWK NoD          0.6350	
QWK Dist	       0.6709
mAP NoD          0.5013	
mAP Dist         0.5258	
Tuned QWK NoD    0.6911	
Tuned QWK Dist   0.7121

		

The best result came from Experiment 6, combining strong augmentation with threshold tuning, yielding 0.6709 test
QWK (0.7121 after tuning). Over-correcting for class imbalance using random crops, weighted sampling and focal
loss hurt the primary metric by pushing predictions towards rare grades. The findings emphasise that balanced
preprocessing, moderate augmentation, careful optimisation and an appropriate choice of metric are more beneficial than
aggressive imbalance correction. 

Reference:"Training data-efficient image transformers & distillation through attention" (Touvron et al., ICML 2021)

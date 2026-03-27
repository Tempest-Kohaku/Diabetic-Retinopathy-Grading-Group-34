# Diabetic-Retinopathy-Grading-Group-34
Diabetic Retinopathy Grading using Deep Learning (PyTorch): A machine learning project that classifies retinal fundus images into diabetic retinopathy severity levels using CNN and Vision Transformer models trained on the Kaggle DR dataset.

Simanta 

DeiT(Data-Efficient Image Transformer):

I am using the Data-Efficient Image Transformer(DeiT-small) model which has 22.1M parameters.Initially I have compressed the images to 224 * 224 *3 and JPEG Compression at Quality at 75 by taking 3000 image per chunks (tried doing 500, 1000 and 2000)[used Kaggle for this step].The dataset then split  into train 60%, val 20% and test 20%sets.After that for data augmentation , i did the Horizontal Flip(50%),rotation 15° and Randomly adjust brightness & contrast (ColorJitter 20%) on train set.For normalization , i used the ImageNet.The DeiT-small model uses images as patches and multi-head self-attention.In training  CrossEntropy loss is used.Moreover , AdamW optimizer, LR (5e-5) for fine-tuning, and cosine annealing for smooth learning rate decay are being used.I am still working on the evaluation part .

Reference:"Training data-efficient image transformers & distillation through attention" (Touvron et al., ICML 2021)

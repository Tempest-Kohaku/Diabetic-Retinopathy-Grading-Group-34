# Diabetic-Retinopathy-Grading-Group-34

Bilal--Resnet50

I am resizing images in the dataset to 224x224x3 (aspect ratio 1:1). I will then resize to 224x149x3 and add zero padding to make it 224x224x3. This way the original aspect ratio of 1.5:1 will be maintained. I will train and evaluate the model seperately with both these resizing techniques to see which technique performs better. Afterwards, I am planning to increase the resolution of the images to maybe 384x384x3 to see if increase in resolution affects performance (I would need to change the resnet-50 pooling layer for this). 

I will also try normalization using ImageNet standard mean and SD, as well as normalization using mean and SD calculated from this particuolar dataset. I will check which technique provides better performance

I am planning to use fine tunned Resnet-50 for this classification task. I will be modifying the classifier head to integrate a final dense layer with 5 units with cross entropy loss which will hopefully fulfill the requirements of this task.

Since the dataset is not balanced i.e 73%, 7%, 15%, 2%, and 2% images belong to class 0, 1, 2, 3, 4. I will try using weighted loss function to see its effect.

I will first train the model without using any data augmentation and without fixing the horizontal flip problem in kaggle`s dataset. I will evaluate the performance of the model and the training patter. Afterwards, I will try to fix the horizontal flip problem in the dataset and will try to add a preprocessing technique that detects the disoriented images and flip them to keep all the images in the dataset in the same orientation. I will increase the colour contrast in the images as well which I think would be beneficial. I will test different augmentation techniques. Then I will do the hyperparameter tunning and test the performance on different values of hyperparameters and also with different augmentation technique.

If everything goes well then I will try to implement the improved pooling function mentioned in this paper: Bhimavarapu, U., Chintalapudi, N. and Battineni, G., 2023. Automatic detection and classification of diabetic retinopathy using the improved pooling function in the convolution neural network. Diagnostics, 13(15), p.2606. Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC10416913/

I also want to implement DINOv2 after all this if I have time.

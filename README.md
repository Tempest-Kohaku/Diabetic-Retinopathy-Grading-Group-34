# Diabetic-Retinopathy-Grading-Group-34

Bilal--Resnet50

I am planning to use fine tunned Resnet-50 for this classification task. I will be modifying the classifier head to integrate a final dense layer with 5 units and softmax activition which will hopefully fulfill the requirements of this task.

I will first train the model without using any data augmentation and without fixing the horizontal flip problem in kaggle`s dataset. I will evaluate the performance of the model and the training patter. Afterwards, I will try to fix the horizontal flip problem in the dataset and will try to add a preprocessing technique that detects the misoriented images and flip them to keep all the images in the dataset in the same orientation. I will increase the colour contrast in the images as well which I think would be beneficial. I will test different augmentation techniques. Then I will do the hyperparameter tunning and test the performance on different values of hyperparameters and also with different augmentation technique.

If everything goes well then I will try to implement the improved pooling function mentioned in this paper: Bhimavarapu, U., Chintalapudi, N. and Battineni, G., 2023. Automatic detection and classification of diabetic retinopathy using the improved pooling function in the convolution neural network. Diagnostics, 13(15), p.2606. Available at: https://pmc.ncbi.nlm.nih.gov/articles/PMC10416913/

I also want to implement DINOv2 after all this if I have time.

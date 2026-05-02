# Diabetic-Retinopathy-Grading-Group-34
Diabetic Retinopathy Grading using Deep Learning (PyTorch): A machine learning project that classifies retinal fundus images into diabetic retinopathy severity levels using CNN and Vision Transformer models trained on the Kaggle DR dataset.

## Best Experiment:

The best experiment is `logs/run_20260502_113025`. This was an evaluation-only run that reused the checkpoint trained in `logs/run_20260501_082822`:

```text
models/best_efficientnet_b5_20260501_082822.pt
```

The model was an ImageNet-pretrained EfficientNet-B5 trained as a single-output ordinal regression model. Instead of predicting five independent class logits, it predicted one continuous diabetic retinopathy severity score in the range of the five grades. The continuous output was then converted into class labels `0` to `4`.

Training details for the checkpoint:

- Model: EfficientNet-B5 from `timm`
- Train/validation/test split sizes: `24585 / 3513 / 7024` or `70% / 10% / 20%`
- Task: single-output regression
- Loss: sample-weighted MSE for training, plain MSE for validation/test
- Optimizer: AdamW
- Batch size: `16`
- Gradient accumulation: `4`
- Effective batch size: `64`
- Learning rate: `2e-4`
- Weight decay: `1e-4`
- Scheduler: 2-epoch linear warmup followed by cosine annealing to `1e-6`
- Best validation QWK before threshold tuning: `0.831571`

The main improvement in the best experiment came from tuning the ordinal thresholds on the validation set.
The default rounding thresholds were:

```text
[0.5, 1.5, 2.5, 3.5]
```

The tuned thresholds were:

```text
[0.76, 1.46, 2.32, 3.26]
```

This improved validation QWK from `0.831571` to `0.844651`. <br>
On the test set, tuned thresholds improved QWK from `0.813670` to `0.824061`.

The final best result used 16-pass test-time augmentation (TTA). Each test image was augmented multiple times using the training augmentations, the model's continuous outputs were averaged, and the tuned thresholds were applied to the averaged output.

Final test performance:
- Base Test Accuracy: `0.807802`
- Base Test QWK: `0.813670`
- Test QWK with tuned thresholds: `0.824061`
- **Test QWK with tuned thresholds with 16-pass TTA: `0.827277`**
- Test accuracy with TTA: `0.806663`
- Test loss with TTA: `0.289215`

TTA gave a smaller improvement than threshold tuning, increasing QWK by `+0.003216`. It slightly reduced exact accuracy, but improved ordinal quality by reducing larger-distance mistakes. 

Overall, the best-performing setup was:

```text
EfficientNet-B5 regression + class-weighted MSE + validation-tuned ordinal thresholds + 16-pass TTA
```

The final best reported test QWK was:

```text
0.827277
```


## My plan using EfficientNetB5:
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Work on preprocessing the data on Kaggle notebook so it reduces the size from 89GBs to something more manageable on my system<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Work on creating stratified train / test / validation splits from the preprocessed images and labels<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Work on handling class imbalance using weighted CrossEntropyLoss<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Work on creating a custom Dataset class, Data Loaders and Data augmentations for training of model<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Work on loading and setting up EfficientNetB3<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Update EfficientNetB3 to EfficientNetB5<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Redo preprocessing since input size is different between the models<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Load and set up EfficientNetB5<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Work on training EfficientNetB5 on the preprocessed dataset<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Visualize training curves (losses, accuracy)<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Log training and test data<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Visualise evaluation<br>
[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Experiment with adding warmup schedulers, changing optimisers, LRs, weight_decays, batch_sizes, etc<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Batch Size = 16, LR = 3e-4, Weight_Decay = 1e-4, AdamW Optimizer, Cosine Scheduling <br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Rotate and flip the test images and average softmax result<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Try accumulating steps to avoid OOM after increasing batch size to 64ish and learning rate to 6e-4ish<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Linear Warmup before Cosine using LR=4e-4 using sequential scheduler<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Add label-smoothing to CrossEntropyLoss<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Use WeightedRandomSampler with unweighted Cross Entropy for sample-based criterion<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Change criterion to treat problem as a single class problem by using MSELoss criterion.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Changing MSELoss to class weighted MSELoss<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Use coordinate ascent to search for four cutoffs that maximises validation QWK. Use those tuned thresholds to check if there is any improvement in QWK compared to the base classification which used rounding.<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[x]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Seeing improvement in tuned thresholds, rerun best experiment(Exp 8 with class weighted MSELoss)<br>


import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------------------
# Configuration
# ---------------------------
DATA_DIR = "path/to/kaggle/input"         
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "train_images")   
IMG_SIZE = 224                                      
NUM_WORKERS = 4
SEED = 42

# ---------------------------
# 1. Custom Dataset Class
# ---------------------------
class DRDataset(Dataset):
    
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # (Optional) Filter out missing images
        self.valid_rows = []
        for idx, row in self.df.iterrows():
            img_path = os.path.join(self.img_dir, row['id_code'] + ".png")
            if os.path.exists(img_path):
                self.valid_rows.append(row)
        self.df = pd.DataFrame(self.valid_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['id_code'] + ".png"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')  
        label = int(row['diagnosis'])

        if self.transform:
            image = self.transform(image)

        return image, label

# ---------------------------
# 2. Transforms (No Augmentation)
# ---------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


train_transform = base_transform
val_transform   = base_transform

# ---------------------------
# 3. Create Dataset Splits
# ---------------------------
full_df = pd.read_csv(TRAIN_CSV)
train_ids, val_ids = train_test_split(
    full_df['id_code'].values,
    test_size=0.2,
    stratify=full_df['diagnosis'],      
    random_state=SEED
)

train_df = full_df[full_df['id_code'].isin(train_ids)]
val_df   = full_df[full_df['id_code'].isin(val_ids)]


train_csv = os.path.join(DATA_DIR, "train_split.csv")
val_csv   = os.path.join(DATA_DIR, "val_split.csv")
train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)

# ---------------------------
# 4. Instantiate Datasets and DataLoaders
# ---------------------------
train_dataset = DRDataset(csv_file=train_csv, img_dir=IMAGE_DIR, transform=train_transform)
val_dataset   = DRDataset(csv_file=val_csv,   img_dir=IMAGE_DIR, transform=val_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ---------------------------
# 5. Quick Test: Iterate One Batch
# ---------------------------
if __name__ == "__main__":
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")   
        print(f"Labels: {labels}")
        break
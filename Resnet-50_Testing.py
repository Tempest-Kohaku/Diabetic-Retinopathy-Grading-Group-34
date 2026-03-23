import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import seaborn as sns

# ===================== CONFIG =====================
model_path = "checkpoints/checkpoint.pth"   # or models/model_epoch_10.pth
test_dir = "test/"
excel_path = "labels.xlsx"

batch_size = 32
log_file = "test_log.txt"
model_name = "ResNet50"

# ===================== FUNCTIONS =====================
def load_labels(path):
    df = pd.read_excel(path)
    return dict(zip(df['filename'], df['label']))

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

class TestDataset(Dataset):
    def __init__(self, folder, label_dict, transform=None):
        self.samples = []
        self.transform = transform

        for fname in os.listdir(folder):
            if fname in label_dict:
                self.samples.append((os.path.join(folder, fname), label_dict[fname]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# ===================== MODEL =====================
def build_model(num_classes, device):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=device)

    loaded_epoch = None

    # Case 1: checkpoint file
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'epoch' in checkpoint:
            loaded_epoch = checkpoint['epoch'] + 1

    # Case 2: normal model file
    else:
        model.load_state_dict(checkpoint)

        # Try extracting epoch from filename
        fname = os.path.basename(model_path)
        if "epoch_" in fname:
            try:
                loaded_epoch = int(fname.split("epoch_")[1].split(".")[0])
            except:
                loaded_epoch = None

    return model.to(device), loaded_epoch

# ===================== TEST =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_dict = load_labels(excel_path)
    num_classes = len(set(label_dict.values()))

    dataset = TestDataset(test_dir, label_dict, get_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model, loaded_epoch = build_model(num_classes, device)
    model.eval()

    preds, labels = [], []

    loop = tqdm(loader, desc="Testing")

    with torch.no_grad():
        for x, y in loop:
            x = x.to(device)
            out = model(x)
            p = torch.argmax(out, dim=1)

            preds.extend(p.cpu().numpy())
            labels.extend(y.numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')

    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

    # ===================== CONFUSION MATRIX =====================
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ===================== LOG =====================
    with open(log_file, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Test path: {test_dir}\n")

        if loaded_epoch is not None:
            f.write(f"Loaded Model Epoch: {loaded_epoch}\n")
        else:
            f.write("Loaded Model Epoch: Unknown\n")

        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

# ===================== RUN =====================
if __name__ == "__main__":
    main()
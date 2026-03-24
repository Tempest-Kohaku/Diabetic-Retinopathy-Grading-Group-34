#======================================================================#
# python3 -m venv venv
#source venv/bin/activate
#pip install torch torchvision matplotlib pandas openpyxl scikit-learn tqdm
#======================================================================#

import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score
from tqdm import tqdm

# ===================== CONFIG =====================
data_dir = "dataset/"
excel_path = "labels.xlsx"

batch_size = 32
learning_rate = 1e-4
num_epochs = 10
model_name = "ResNet50"

# Output folders
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

log_file = "logs/train_log.txt"
checkpoint_path = "checkpoints/checkpoint.pth"

# ===================== DATASET =====================
class ImageDataset(Dataset):
    def __init__(self, data_dir, label_dict, transform=None):
        self.samples = []
        self.transform = transform

        for fname in os.listdir(data_dir):
            if fname in label_dict:
                self.samples.append((os.path.join(data_dir, fname), label_dict[fname]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# ===================== FUNCTIONS =====================
def load_labels(path):
    df = pd.read_excel(path)
    label_dict = dict(zip(df['filename'], df['label']))
    num_classes = len(set(label_dict.values()))
    return label_dict, num_classes

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

def get_loaders(label_dict):
    dataset = ImageDataset(data_dir, label_dict, get_transform())

    train_size = int(0.8235 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def build_model(num_classes, device):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc="Training")
    for x, y in loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = criterion(out, y)

            total_loss += loss.item()

            p = torch.argmax(out, dim=1)
            preds.extend(p.cpu().numpy())
            labels.extend(y.cpu().numpy())

    f1 = f1_score(labels, preds, average='weighted')
    return total_loss / len(loader), f1

def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.savefig("plots/loss_plot.png")
    plt.close()

# ===================== MAIN =====================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_dict, num_classes = load_labels(excel_path)
    train_loader, val_loader = get_loaders(label_dict)

    model = build_model(num_classes, device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    train_losses, val_losses = [], []

    # ===== Resume =====
    if os.path.exists(checkpoint_path):
        print("Resuming training...")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']

        with open(log_file, 'a') as f:
            f.write(f"\n--- Resumed from epoch {start_epoch} ---\n")

    else:
        with open(log_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Batch size: {batch_size}\n")
            f.write(f"Learning rate: {learning_rate}\n")
            f.write(f"Num classes: {num_classes}\n\n")

    # ===== TRAIN LOOP =====
    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f"\nEpoch {epoch+1}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train={train_loss:.4f}, Val={val_loss:.4f}, F1={val_f1:.4f}")

        # Log
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, F1={val_f1:.4f}\n")

        # Save model
        torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}.pth")

        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses
        }, checkpoint_path)

        # Plot
        plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()

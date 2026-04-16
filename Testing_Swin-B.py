# ====================================================================== #
# python3 -m venv venv
# source venv/bin/activate
# pip install torch torchvision matplotlib pandas scikit-learn tqdm timm numpy pillow
# ====================================================================== #

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, cohen_kappa_score
from tqdm import tqdm
import timm

# ===================== CONFIG =====================
EXPERIMENT_NAME = "exp5_wce_mixup"

model_path = "/user/HS401/bs01338/Downloads/DRG/Diabetic-Retinopathy-Grading-Group-34/models/exp5_wce_mixup/model_epoch_010_qwk_0.7643.pth"
test_dir = r"/user/HS401/bs01338/Downloads/DRG Dataset 384/test"
csv_path = r"/user/HS401/bs01338/Downloads/DRG Dataset 384/test.csv"

batch_size = 32
num_classes = 5
model_name = "swin_base_patch4_window12_384"
num_workers = min(4, os.cpu_count() or 1)

log_file = f"test_log_{EXPERIMENT_NAME}.txt"
confusion_matrix_path = f"confusion_matrix_{EXPERIMENT_NAME}.png"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ===================== HELPERS =====================
def check_paths():
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")


def load_labels(path):
    df = pd.read_csv(path)

    if {"image", "level"}.issubset(df.columns):
        image_col, label_col = "image", "level"
    elif {"filename", "class"}.issubset(df.columns):
        image_col, label_col = "filename", "class"
    else:
        raise ValueError("CSV must contain either ('image', 'level') or ('filename', 'class').")

    df[image_col] = df[image_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(int)

    labels = set(df[label_col].unique().tolist())
    if not labels.issubset(set(range(num_classes))):
        raise ValueError(f"Found labels outside 0-{num_classes - 1}: {sorted(labels)}")

    duplicates = df[image_col][df[image_col].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate image names found in CSV. Example duplicates: {duplicates[:10]}")

    return dict(zip(df[image_col], df[label_col]))


def build_samples(folder, label_dict):
    file_map = {}
    duplicates = []

    for fname in os.listdir(folder):
        full_path = os.path.join(folder, fname)
        if not os.path.isfile(full_path):
            continue

        stem, ext = os.path.splitext(fname)
        if ext.lower() not in IMAGE_EXTS:
            continue

        key = stem.strip().lower()
        if key in file_map:
            duplicates.append(key)
        file_map[key] = fname

    if duplicates:
        raise ValueError(
            f"Duplicate image basenames found in test folder. Example duplicates: {duplicates[:10]}"
        )

    samples, missing = [], []
    for name, label in label_dict.items():
        key = name.strip().lower()
        if key in file_map:
            samples.append((os.path.join(folder, file_map[key]), int(label)))
        else:
            missing.append(name)

    print("\n================ TEST FILE MATCHING ================")
    print(f"CSV entries        : {len(label_dict)}")
    print(f"Matched image files: {len(samples)}")
    print(f"Missing image files: {len(missing)}")
    if missing:
        print(f"Example missing files (up to 20): {missing[:20]}")
    print("====================================================\n")

    return samples


def parse_epoch(path):
    match = re.search(r"epoch_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else None


class TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.transform(img), label


def build_model(device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


def save_confusion_matrix(cm, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix ({EXPERIMENT_NAME})")
    plt.colorbar()
    ticks = list(range(num_classes))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_log(path, loaded_epoch, metrics):
    recall_str = ", ".join(f"C{i}={r:.4f}" for i, r in enumerate(metrics["recall_per_class"]))

    with open(path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TEST RUN LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment             : {EXPERIMENT_NAME}\n")
        f.write(f"Model                  : {model_name}\n")
        f.write(f"Loaded Epoch           : {loaded_epoch if loaded_epoch is not None else 'Unknown'}\n")
        f.write(f"Batch Size             : {batch_size}\n")
        f.write("\n")
        f.write("METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test Loss              : {metrics['test_loss']:.4f}\n")
        f.write(f"Accuracy               : {metrics['accuracy']:.4f}\n")
        f.write(f"QWK                    : {metrics['qwk']:.4f}\n")
        f.write(f"Macro F1               : {metrics['macro_f1']:.4f}\n")
        f.write(f"Recall                 : {recall_str}\n")
        f.write(f"Confusion Matrix       : {confusion_matrix_path}\n")


# ===================== MAIN =====================
def main():
    check_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    labels = load_labels(csv_path)
    samples = build_samples(test_dir, labels)
    if not samples:
        raise ValueError("No test samples found. Check filename matching.")

    dataset = TestDataset(samples)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    loader = DataLoader(dataset, **loader_kwargs)

    model = build_model(device)
    model.eval()
    loaded_epoch = parse_epoch(model_path)
    criterion = nn.CrossEntropyLoss()

    preds, targets = [], []
    total_loss = 0.0

    with torch.inference_mode():
        for x, y in tqdm(loader, desc="Testing"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)

            total_loss += loss.item()
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    metrics = {
        "test_loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy_score(targets, preds),
        "qwk": cohen_kappa_score(targets, preds, labels=list(range(num_classes)), weights="quadratic"),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "recall_per_class": recall_score(
            targets, preds, average=None, labels=list(range(num_classes)), zero_division=0
        ),
    }
    metrics["qwk"] = 0.0 if np.isnan(metrics["qwk"]) else metrics["qwk"]

    print(f"\nExperiment: {EXPERIMENT_NAME}")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"QWK      : {metrics['qwk']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print("Recall   : " + ", ".join(f"C{i}={r:.4f}" for i, r in enumerate(metrics["recall_per_class"])))

    save_confusion_matrix(cm, confusion_matrix_path)
    write_log(log_file, loaded_epoch, metrics)


if __name__ == "__main__":
    main()
# ====================================================================== #
# python3 -m venv venv
# source venv/bin/activate
# pip install torch torchvision matplotlib pandas scikit-learn tqdm timm numpy pillow coral-pytorch
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

from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits


# ===================== CONFIG =====================
EXPERIMENT_NAME = "exp_parallel_effb4_swinb384_corn_fusion_head_test"

# CHANGE THIS TO YOUR ACTUAL .pth MODEL FILE
model_path = "/user/HS401/bs01338/Downloads/DRG/Diabetic-Retinopathy-Grading-Group-34/models/exp_parallel_effb4_swinb384_corn_fusion_head/model_epoch_011_qwk_0.8127.pth"

test_dir = r"/user/HS401/bs01338/Downloads/CLAHE/Test"
csv_path = r"/user/HS401/bs01338/Downloads/DRG Dataset 384/test.csv"

batch_size = 32
num_classes = 5
num_workers = min(4, os.cpu_count() or 1)

# Must match training script exactly
efficientnet_name = "efficientnet_b4.ra2_in1k"
swin_name = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"
model_name = "parallel_efficientnet_b4_swin_base_384"

fusion_hidden_dim = 1024
fusion_dropout = 0.3

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


# ===================== MODEL =====================
class ParallelEfficientNetSwinCORN(nn.Module):
    def __init__(
        self,
        efficientnet_name,
        swin_name,
        num_classes,
        fusion_hidden_dim=1024,
        fusion_dropout=0.3,
        pretrained=False,
    ):
        super().__init__()

        self.efficientnet = timm.create_model(
            efficientnet_name,
            pretrained=pretrained,
            num_classes=0,
        )
        self.swin = timm.create_model(
            swin_name,
            pretrained=pretrained,
            num_classes=0,
        )

        eff_dim = getattr(self.efficientnet, "num_features", None)
        swin_dim = getattr(self.swin, "num_features", None)
        if eff_dim is None or swin_dim is None:
            raise ValueError("Could not determine backbone feature dimensions from timm models.")

        fusion_dim = eff_dim + swin_dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, num_classes - 1),
        )

    @staticmethod
    def _to_vector(feat):
        if feat.ndim == 2:
            return feat
        if feat.ndim == 3:
            return feat.mean(dim=1)
        if feat.ndim == 4:
            if feat.shape[1] > feat.shape[-1] and feat.shape[1] > feat.shape[-2]:
                return feat.mean(dim=(2, 3))
            return feat.mean(dim=(1, 2))
        raise ValueError(f"Unexpected feature shape: {feat.shape}")

    def forward(self, x):
        eff_feat = self._to_vector(self.efficientnet(x))
        swin_feat = self._to_vector(self.swin(x))
        fused = torch.cat([eff_feat, swin_feat], dim=1)
        return self.fusion(fused)


def clean_state_dict(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def build_model(device):
    model = ParallelEfficientNetSwinCORN(
        efficientnet_name=efficientnet_name,
        swin_name=swin_name,
        num_classes=num_classes,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        pretrained=False,
    )

    state_dict = torch.load(model_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise ValueError("Loaded .pth file is not a valid state_dict dictionary.")

    state_dict = clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)

    return model.to(device)


# ===================== OUTPUTS =====================
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
    f1_per_class_str = ", ".join(f"C{i}={v:.4f}" for i, v in enumerate(metrics["f1_per_class"]))

    with open(path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TEST RUN LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Experiment             : {EXPERIMENT_NAME}\n")
        f.write(f"Model                  : {model_name}\n")
        f.write(f"Branch 1               : {efficientnet_name}\n")
        f.write(f"Branch 2               : {swin_name}\n")
        f.write("Fusion type            : Parallel feature fusion with MLP fusion head\n")
        f.write(f"Fusion hidden dim      : {fusion_hidden_dim}\n")
        f.write(f"Fusion dropout         : {fusion_dropout}\n")
        f.write("Fusion head            : LayerNorm -> Linear -> GELU -> Dropout -> Linear(num_classes - 1)\n")
        f.write("Ordinal method         : CORN\n")
        f.write(f"Loaded Epoch           : {loaded_epoch if loaded_epoch is not None else 'Unknown'}\n")
        f.write(f"Batch Size             : {batch_size}\n")
        f.write("\n")
        f.write("METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test CORN Loss         : {metrics['test_corn_loss']:.4f}\n")
        f.write(f"Accuracy               : {metrics['accuracy']:.4f}\n")
        f.write(f"QWK                    : {metrics['qwk']:.4f}\n")
        f.write(f"Micro F1               : {metrics['micro_f1']:.4f}\n")
        f.write(f"Weighted F1            : {metrics['weighted_f1']:.4f}\n")
        f.write(f"Macro F1               : {metrics['macro_f1']:.4f}\n")
        f.write(f"Per-class F1           : {f1_per_class_str}\n")
        f.write(f"Recall                 : {recall_str}\n")
        f.write(f"Confusion Matrix       : {confusion_matrix_path}\n")


# ===================== MAIN =====================
def main():
    check_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Using device: {device}")
    print(f"Model path: {model_path}")

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

    preds, targets = [], []
    total_loss = 0.0

    with torch.inference_mode():
        for x, y in tqdm(loader, desc="Testing"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)

            loss = corn_loss(out.float(), y, num_classes=num_classes)
            total_loss += loss.item()

            batch_preds = corn_label_from_logits(out.float()).cpu().numpy()
            preds.extend(batch_preds)
            targets.extend(y.cpu().numpy())

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    metrics = {
        "test_corn_loss": total_loss / max(1, len(loader)),
        "accuracy": accuracy_score(targets, preds),
        "qwk": cohen_kappa_score(targets, preds, labels=list(range(num_classes)), weights="quadratic"),
        "micro_f1": f1_score(targets, preds, average="micro", zero_division=0),
        "weighted_f1": f1_score(targets, preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "f1_per_class": f1_score(
            targets, preds, average=None, labels=list(range(num_classes)), zero_division=0
        ),
        "recall_per_class": recall_score(
            targets, preds, average=None, labels=list(range(num_classes)), zero_division=0
        ),
    }
    metrics["qwk"] = 0.0 if np.isnan(metrics["qwk"]) else metrics["qwk"]

    print(f"\nExperiment: {EXPERIMENT_NAME}")
    print(f"Test CORN Loss: {metrics['test_corn_loss']:.4f}")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"QWK           : {metrics['qwk']:.4f}")
    print(f"Micro F1      : {metrics['micro_f1']:.4f}")
    print(f"Weighted F1   : {metrics['weighted_f1']:.4f}")
    print(f"Macro F1      : {metrics['macro_f1']:.4f}")
    print("Per-class F1  : " + ", ".join(f"C{i}={v:.4f}" for i, v in enumerate(metrics["f1_per_class"])))
    print("Recall        : " + ", ".join(f"C{i}={r:.4f}" for i, r in enumerate(metrics["recall_per_class"])))

    save_confusion_matrix(cm, confusion_matrix_path)
    write_log(log_file, loaded_epoch, metrics)

    print(f"Confusion matrix saved to: {confusion_matrix_path}")
    print(f"Test log saved to: {log_file}")


if __name__ == "__main__":
    main()
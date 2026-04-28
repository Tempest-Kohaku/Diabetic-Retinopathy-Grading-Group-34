# ====================================================================== #
# python3 -m venv venv
# source venv/bin/activate
# pip install torch torchvision pandas scikit-learn tqdm timm numpy pillow matplotlib coral-pytorch
# ====================================================================== #

import os
import random
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, cohen_kappa_score
from tqdm import tqdm
import timm

from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits


# ===================== CONFIG =====================
SEED = 3
EXPERIMENT_TAG = "exp_parallel_effb4_swinb384_corn_fusion_head"

# Keep fixed across experiments if you want the same split
SPLIT_TAG = "dr_fixed_split_v1"

data_dir = r"/user/HS401/bs01338/Downloads/CLAHE/Train"
excel_path = r"/user/HS401/bs01338/Downloads/DRG Dataset 384/train.csv"

# Kept unchanged
batch_size = 14

learning_rate = 3e-5
weight_decay = 1e-4
num_epochs = 20
num_classes = 5

efficientnet_name = "efficientnet_b4.ra2_in1k"
swin_name = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"
model_name = "parallel_efficientnet_b4_swin_base_384"

# Fusion head settings
fusion_hidden_dim = 1024
fusion_dropout = 0.3

top_k_models = 2
enable_grad_checkpointing = True

num_workers = min(4, os.cpu_count() or 1)

logs_dir = "logs"
splits_dir = "splits"
plots_dir = "plots"
models_dir = os.path.join("models", EXPERIMENT_TAG)

log_file = os.path.join(logs_dir, f"train_log_{EXPERIMENT_TAG}.txt")
history_csv_path = os.path.join(logs_dir, f"metrics_history_{EXPERIMENT_TAG}.csv")
train_split_path = os.path.join(splits_dir, f"train_split_{SPLIT_TAG}.csv")
val_split_path = os.path.join(splits_dir, f"val_split_{SPLIT_TAG}.csv")

loss_plot_path = os.path.join(plots_dir, f"train_val_loss_{EXPERIMENT_TAG}.png")
recall_plot_path = os.path.join(plots_dir, f"per_class_recall_{EXPERIMENT_TAG}.png")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

for d in (logs_dir, models_dir, splits_dir, plots_dir):
    os.makedirs(d, exist_ok=True)


# ===================== REPRODUCIBILITY =====================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_torch_generator(seed=SEED):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def check_startup_paths():
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"CSV file not found: {excel_path}")


def check_disk_space(path=".", min_free_gb=2.0):
    free_bytes = shutil.disk_usage(path).free
    free_gb = free_bytes / (1024 ** 3)
    print(f"Free disk space: {free_gb:.2f} GB")
    if free_gb < min_free_gb:
        print(f"[Warning] Low disk space. Less than {min_free_gb:.1f} GB free.")


set_seed(SEED)


# ===================== TRANSFORMS =====================
def build_train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ===================== DATASET =====================
class ImageDataset(Dataset):
    def __init__(self, samples, train=False):
        self.samples = samples
        self.transform = build_train_transform() if train else build_eval_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        with Image.open(path) as img:
            img = img.convert("RGB")

        img = self.transform(img)
        return img, int(label)


# ===================== DATA =====================
def load_labels(path):
    df = pd.read_csv(path)

    required_cols = {"image", "level"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    df["image"] = df["image"].astype(str).str.strip()
    df["level"] = df["level"].astype(int)

    unique_labels = set(df["level"].unique().tolist())
    valid_labels = set(range(num_classes))
    if not unique_labels.issubset(valid_labels):
        raise ValueError(f"Found labels outside 0-{num_classes - 1}: {sorted(unique_labels)}")

    duplicates = df["image"][df["image"].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate image names found in CSV. Example duplicates: {duplicates[:10]}")

    return dict(zip(df["image"], df["level"]))


def build_file_mapping():
    mapping, duplicates = {}, []

    for f in os.listdir(data_dir):
        full_path = os.path.join(data_dir, f)
        if not os.path.isfile(full_path):
            continue

        base, ext = os.path.splitext(f)
        if ext.lower() not in IMAGE_EXTS:
            continue

        key = base.strip().lower()
        if key in mapping:
            duplicates.append(key)
        mapping[key] = f

    if duplicates:
        raise ValueError(
            "Duplicate image basenames found in folder. "
            f"Example duplicates: {duplicates[:10]}"
        )

    return mapping


def get_samples(label_dict):
    file_map = build_file_mapping()
    samples, missing_files = [], []

    for fname, label in label_dict.items():
        key = fname.strip().lower()
        if key in file_map:
            samples.append((os.path.join(data_dir, file_map[key]), int(label)))
        else:
            missing_files.append(fname)

    print("\n================ FILE MATCHING ================")
    print(f"CSV entries        : {len(label_dict)}")
    print(f"Matched image files: {len(samples)}")
    print(f"Missing image files: {len(missing_files)}")
    if missing_files:
        print(f"Example missing files (up to 20): {missing_files[:20]}")
    print("==============================================\n")

    return samples


def compute_class_stats(labels):
    return pd.Series(labels).value_counts().reindex(range(num_classes), fill_value=0).astype(int)


def save_split_manifest(train_samples, val_samples):
    pd.DataFrame(train_samples, columns=["path", "label"]).to_csv(train_split_path, index=False)
    pd.DataFrame(val_samples, columns=["path", "label"]).to_csv(val_split_path, index=False)


def load_split_manifest():
    train_df = pd.read_csv(train_split_path)
    val_df = pd.read_csv(val_split_path)

    required_cols = {"path", "label"}
    if not required_cols.issubset(train_df.columns) or not required_cols.issubset(val_df.columns):
        raise ValueError("Split CSV files must contain 'path' and 'label' columns.")

    train_samples = list(zip(train_df["path"].astype(str), train_df["label"].astype(int)))
    val_samples = list(zip(val_df["path"].astype(str), val_df["label"].astype(int)))
    return train_samples, val_samples


# ===================== LOADERS =====================
def get_loaders(label_dict, device):
    samples = get_samples(label_dict)
    if not samples:
        raise ValueError("No matched samples found. Please check data_dir and CSV image names.")

    use_saved_split = os.path.isfile(train_split_path) and os.path.isfile(val_split_path)

    if use_saved_split:
        train_samples, val_samples = load_split_manifest()
        print("Using existing train/validation split manifests.")
    else:
        paths = [path for path, _ in samples]
        labels = [label for _, label in samples]

        try:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                paths, labels, test_size=0.17, stratify=labels, random_state=SEED
            )
        except ValueError as e:
            raise ValueError(
                "train_test_split failed. This usually happens when one or more classes "
                f"have too few samples for stratification. Original error: {e}"
            )

        train_samples = list(zip(train_paths, train_labels))
        val_samples = list(zip(val_paths, val_labels))
        save_split_manifest(train_samples, val_samples)
        print("Created and saved new train/validation split manifests.")

    train_labels = [label for _, label in train_samples]
    val_labels = [label for _, label in val_samples]

    train_ds = ImageDataset(train_samples, train=True)
    val_ds = ImageDataset(val_samples, train=False)

    train_class_counts = compute_class_stats(train_labels)
    val_class_counts = compute_class_stats(val_labels)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        common_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=build_torch_generator(SEED),
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **common_kwargs,
    )

    return train_loader, val_loader, train_class_counts, val_class_counts, len(train_samples)


# ===================== MODEL =====================
class ParallelEfficientNetSwinCORN(nn.Module):
    def __init__(
        self,
        efficientnet_name,
        swin_name,
        num_classes,
        fusion_hidden_dim=1024,
        fusion_dropout=0.3,
        pretrained=True,
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


def build_model(device):
    model = ParallelEfficientNetSwinCORN(
        efficientnet_name=efficientnet_name,
        swin_name=swin_name,
        num_classes=num_classes,
        fusion_hidden_dim=fusion_hidden_dim,
        fusion_dropout=fusion_dropout,
        pretrained=True,
    )

    if enable_grad_checkpointing:
        if hasattr(model.efficientnet, "set_grad_checkpointing"):
            model.efficientnet.set_grad_checkpointing(True)
        if hasattr(model.swin, "set_grad_checkpointing"):
            model.swin.set_grad_checkpointing(True)

    return model.to(device)


# ===================== LOGGING =====================
def initialize_log_file(
    log_path,
    train_class_counts,
    val_class_counts,
    num_train_samples,
):
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING RUN LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Run started            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment             : {EXPERIMENT_TAG}\n")
        f.write(f"Split tag              : {SPLIT_TAG}\n")
        f.write(f"Model                  : {model_name}\n")
        f.write(f"Branch 1               : {efficientnet_name}\n")
        f.write(f"Branch 2               : {swin_name}\n")
        f.write("Fusion type            : Parallel feature fusion with MLP fusion head\n")
        f.write(f"Fusion hidden dim      : {fusion_hidden_dim}\n")
        f.write(f"Fusion dropout         : {fusion_dropout}\n")
        f.write("Fusion head            : LayerNorm -> Linear -> GELU -> Dropout -> Linear(num_classes - 1)\n")
        f.write(f"Num classes            : {num_classes}\n")
        f.write(f"Batch size             : {batch_size}\n")
        f.write(f"Learning rate          : {learning_rate}\n")
        f.write(f"Weight decay           : {weight_decay}\n")
        f.write(f"Num epochs             : {num_epochs}\n")
        f.write("Optimizer              : AdamW\n")
        f.write("Train loss             : CORN loss\n")
        f.write("Validation loss        : CORN loss\n")
        f.write("MixUp                  : Disabled\n")
        f.write("Imbalance handling     : None\n")
        f.write(f"Top model saving       : Top {top_k_models} by validation QWK\n")
        f.write(f"Grad checkpointing     : {'Enabled' if enable_grad_checkpointing else 'Disabled'}\n")
        f.write("Checkpoint saving      : Disabled\n")
        f.write("Training resume        : Disabled\n")
        f.write("Plots                  : CORN loss curve + per-class recall curve\n")
        f.write(f"Seed                   : {SEED}\n\n")

        f.write("IMPORTANT TECHNIQUES USED\n")
        f.write("-" * 80 + "\n")
        f.write("Parallel fusion        : EfficientNet-B4 embedding + Swin-B-384 embedding -> concatenation -> fusion head\n")
        f.write("Train augmentation     : RandomHorizontalFlip, ColorJitter(brightness=0.2, contrast=0.2), RandomAffine(translate=(0.05,0.05))\n")
        f.write("Ordinal method         : CORN (4 logits for 5 classes)\n")
        f.write("Train sampler          : Standard shuffled batches\n")
        f.write(f"Train samples          : {num_train_samples}\n")
        f.write("Loss function          : CORN loss\n")
        f.write("Metrics                : Train CORN loss, Validation CORN loss, QWK, Macro F1, per-class recall\n\n")

        f.write("CLASS STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train class counts     : {train_class_counts.to_dict()}\n")
        f.write(f"Val class counts       : {val_class_counts.to_dict()}\n\n")

        f.write("FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train split path       : {train_split_path}\n")
        f.write(f"Val split path         : {val_split_path}\n")
        f.write(f"Metrics history path   : {history_csv_path}\n")
        f.write(f"Loss plot path         : {loss_plot_path}\n")
        f.write(f"Recall plot path       : {recall_plot_path}\n")
        f.write(f"Models directory       : {models_dir}\n\n")

        f.write("EPOCH METRICS\n")
        f.write("-" * 80 + "\n")


def log_epoch(log_path, epoch, train_loss, metrics, lr):
    recall_str = ", ".join(f"class_{i}={r:.4f}" for i, r in enumerate(metrics["recall_per_class"]))
    with open(log_path, "a") as f:
        f.write(
            f"Epoch {epoch:02d} | "
            f"LR_used={lr:.8f} | "
            f"TrainCORNLoss={train_loss:.4f} | "
            f"ValCORNLoss={metrics['val_corn_loss']:.4f} | "
            f"QWK={metrics['qwk']:.4f} | "
            f"MacroF1={metrics['macro_f1']:.4f} | "
            f"Recall[{recall_str}]\n"
        )


def log_top_models_summary(log_path, top_models):
    with open(log_path, "a") as f:
        f.write("\n")
        f.write(f"TOP {top_k_models} SAVED MODELS\n")
        f.write("-" * 80 + "\n")
        if not top_models:
            f.write("No top models were saved.\n")
            return
        for rank, item in enumerate(sort_top_models(top_models), start=1):
            f.write(
                f"Rank {rank}: Epoch={item['epoch']}, "
                f"QWK={item['qwk']:.4f}, "
                f"Path={item['path']}\n"
            )


# ===================== HISTORY / PLOTTING =====================
def history_columns():
    return ["epoch", "train_loss", "val_corn_loss", "qwk", "macro_f1"] + [
        f"recall_class_{i}" for i in range(num_classes)
    ]


def empty_history_df():
    return pd.DataFrame(columns=history_columns())


def reset_history_artifacts():
    for path in (history_csv_path, loss_plot_path, recall_plot_path):
        if os.path.isfile(path):
            os.remove(path)


def upsert_history_row(history_df, epoch, train_loss, metrics):
    row = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_corn_loss": float(metrics["val_corn_loss"]),
        "qwk": float(metrics["qwk"]),
        "macro_f1": float(metrics["macro_f1"]),
    }
    for i, value in enumerate(metrics["recall_per_class"]):
        row[f"recall_class_{i}"] = float(value)

    history_df = history_df[history_df["epoch"] != epoch]
    history_df = pd.concat([history_df, pd.DataFrame([row])], ignore_index=True)
    return history_df.sort_values("epoch").reset_index(drop=True)


def save_history_csv(history_df):
    history_df.to_csv(history_csv_path, index=False)


def plot_loss_curve(history_df):
    if history_df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train CORN Loss")
    plt.plot(history_df["epoch"], history_df["val_corn_loss"], marker="o", label="Validation CORN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training CORN Loss vs Validation CORN Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_recall_curve(history_df):
    if history_df.empty:
        return

    plt.figure(figsize=(8, 5))
    for i in range(num_classes):
        col = f"recall_class_{i}"
        plt.plot(history_df["epoch"], history_df[col], marker="o", label=f"Class {i}")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Per-Class Recall vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(recall_plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def update_history_and_plots(history_df, epoch, train_loss, metrics):
    history_df = upsert_history_row(history_df, epoch, train_loss, metrics)
    save_history_csv(history_df)
    plot_loss_curve(history_df)
    plot_recall_curve(history_df)
    return history_df


# ===================== TRAIN / EVAL =====================
def train_epoch(model, loader, optimizer, device, scaler, use_amp):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc="Training", leave=False)

    for x, y in loop:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(device_type=device.type, enabled=use_amp):
            out = model(x)

        loss = corn_loss(out.float(), y, num_classes=num_classes)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(loader))


def evaluate(model, loader, device, use_amp):
    model.eval()
    total_corn_loss = 0.0
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)

            loss = corn_loss(out.float(), y, num_classes=num_classes)
            total_corn_loss += loss.item()

            batch_preds = corn_label_from_logits(out.float()).cpu().numpy()
            preds.extend(batch_preds)
            labels.extend(y.cpu().numpy())

    qwk = cohen_kappa_score(labels, preds, labels=list(range(num_classes)), weights="quadratic")
    qwk = 0.0 if np.isnan(qwk) else qwk

    return {
        "val_corn_loss": total_corn_loss / max(1, len(loader)),
        "qwk": qwk,
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "recall_per_class": recall_score(
            labels, preds, average=None, labels=list(range(num_classes)), zero_division=0
        ),
    }


# ===================== TOP-K MODEL SAVING =====================
def sort_top_models(top_models):
    return sorted(top_models, key=lambda x: (x["qwk"], x["epoch"]), reverse=True)


def update_top_models(model, epoch, qwk, top_models):
    top_models = sort_top_models(top_models)

    qualifies = (
        len(top_models) < top_k_models or
        (qwk, epoch) > (top_models[-1]["qwk"], top_models[-1]["epoch"])
    )

    if not qualifies:
        return top_models, None, []

    save_path = os.path.join(models_dir, f"model_epoch_{epoch:03d}_qwk_{qwk:.4f}.pth")
    tmp_path = save_path + ".tmp"

    try:
        torch.save(model.state_dict(), tmp_path)
        os.replace(tmp_path, save_path)
    except Exception as e:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            f"Failed to save model to: {save_path}\n"
            f"Likely disk space / quota / write issue.\n"
            f"Original error: {e}"
        ) from e

    top_models.append({
        "epoch": epoch,
        "qwk": float(qwk),
        "path": save_path,
    })
    top_models = sort_top_models(top_models)

    removed_paths = []
    while len(top_models) > top_k_models:
        removed_item = top_models.pop(-1)
        removed_path = removed_item["path"]
        if removed_path != save_path and os.path.isfile(removed_path):
            os.remove(removed_path)
            removed_paths.append(removed_path)

    return top_models, save_path, removed_paths


def cleanup_stale_top_model_files(top_models):
    keep_paths = {os.path.abspath(item["path"]) for item in top_models}
    for fname in os.listdir(models_dir):
        if not (fname.endswith(".pth") or fname.endswith(".tmp")):
            continue
        path = os.path.abspath(os.path.join(models_dir, fname))
        if path not in keep_paths and os.path.isfile(path):
            os.remove(path)


# ===================== MAIN =====================
def main():
    check_startup_paths()
    check_disk_space(path=".", min_free_gb=2.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Using device: {device}")
    print(f"Log file: {log_file}")

    label_dict = load_labels(excel_path)
    (
        train_loader,
        val_loader,
        train_class_counts,
        val_class_counts,
        num_train_samples,
    ) = get_loaders(label_dict, device)

    model = build_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = amp.GradScaler(device.type, enabled=use_amp)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    initialize_log_file(
        log_file,
        train_class_counts,
        val_class_counts,
        num_train_samples,
    )
    reset_history_artifacts()
    cleanup_stale_top_model_files([])

    top_models = []
    history_df = empty_history_df()
    last_metrics = None

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, device, scaler, use_amp)
        metrics = evaluate(model, val_loader, device, use_amp)
        last_metrics = metrics
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train CORN Loss={train_loss:.4f} | "
            f"Val CORN Loss={metrics['val_corn_loss']:.4f} | "
            f"QWK={metrics['qwk']:.4f} | "
            f"Macro F1={metrics['macro_f1']:.4f}"
        )
        print(
            "Per-class Recall: "
            + ", ".join(f"class_{i}={r:.4f}" for i, r in enumerate(metrics["recall_per_class"]))
        )

        log_epoch(log_file, epoch, train_loss, metrics, current_lr)
        history_df = update_history_and_plots(history_df, epoch, train_loss, metrics)

        top_models, saved_path, removed_paths = update_top_models(
            model=model,
            epoch=epoch,
            qwk=metrics["qwk"],
            top_models=top_models,
        )

        if saved_path is not None:
            print(f"Saved top model: {saved_path}")
        for removed_path in removed_paths:
            print(f"Removed lower-ranked model: {removed_path}")

        scheduler.step()

    with open(log_file, "a") as f:
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training completed at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if last_metrics is not None:
            f.write(f"Final Val CORN Loss     : {last_metrics['val_corn_loss']:.4f}\n")
            f.write(f"Final QWK               : {last_metrics['qwk']:.4f}\n")
            f.write(f"Final Macro F1          : {last_metrics['macro_f1']:.4f}\n")
            recall_str = ", ".join(f"class_{i}={r:.4f}" for i, r in enumerate(last_metrics["recall_per_class"]))
            f.write(f"Final Recall            : {recall_str}\n")
            f.write(f"Loss Plot               : {loss_plot_path}\n")
            f.write(f"Recall Plot             : {recall_plot_path}\n")

    log_top_models_summary(log_file, top_models)

    print("\nTraining complete.")
    if top_models:
        print(f"Top {top_k_models} models kept in: {models_dir}")
        for rank, item in enumerate(sort_top_models(top_models), start=1):
            print(f"Rank {rank}: epoch={item['epoch']}, qwk={item['qwk']:.4f}, path={item['path']}")
    print(f"Train split saved to: {train_split_path}")
    print(f"Val split saved to: {val_split_path}")
    print(f"Metrics history saved to: {history_csv_path}")
    print(f"Loss plot saved to: {loss_plot_path}")
    print(f"Recall plot saved to: {recall_plot_path}")
    if last_metrics is not None:
        print(f"Final QWK: {last_metrics['qwk']:.4f}")
    print(f"Metrics logged to: {log_file}")


if __name__ == "__main__":
    main()
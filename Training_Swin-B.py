# ====================================================================== #
# python3 -m venv venv
# source venv/bin/activate
# pip install torch torchvision pandas scikit-learn tqdm timm numpy pillow matplotlib
# ====================================================================== #

import os
import glob
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


# ===================== CONFIG =====================
SEED = 3
EXPERIMENT_TAG = "exp5_wce_mixup_alpha_0.3"

data_dir = r"/user/HS401/bs01338/Downloads/DRG Dataset 384/train"
excel_path = r"/user/HS401/bs01338/Downloads/DRG Dataset 384/train.csv"

batch_size = 15
learning_rate = 3e-5
num_epochs = 20
num_classes = 5
model_name = "swin_base_patch4_window12_384"

mixup_alpha = 0.3
mixup_prob = 1.0

num_workers = min(4, os.cpu_count() or 1)

logs_dir = "logs"
checkpoints_dir = "checkpoints"
splits_dir = "splits"
plots_dir = "plots"
models_dir = os.path.join("models", EXPERIMENT_TAG)

log_file = os.path.join(logs_dir, f"train_log_{EXPERIMENT_TAG}.txt")
history_csv_path = os.path.join(logs_dir, f"metrics_history_{EXPERIMENT_TAG}.csv")
checkpoint_path = os.path.join(checkpoints_dir, f"checkpoint_{EXPERIMENT_TAG}.pth")
train_split_path = os.path.join(splits_dir, "train_split.csv")
val_split_path = os.path.join(splits_dir, "val_split.csv")

top_k_models = 3
loss_plot_path = os.path.join(plots_dir, f"train_val_loss_{EXPERIMENT_TAG}.png")
recall_plot_path = os.path.join(plots_dir, f"per_class_recall_{EXPERIMENT_TAG}.png")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

for d in (logs_dir, checkpoints_dir, models_dir, splits_dir, plots_dir):
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


# ===================== DATASET =====================
class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return (self.transform(img) if self.transform else img), label


# ===================== TRANSFORMS =====================
def build_transform(train=False):
    ops = []
    if train:
        ops.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transforms.Compose(ops)


# ===================== MIXUP =====================
def mixup_batch(x, y, alpha=0.2):
    if alpha <= 0 or x.size(0) < 2:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, float(lam)


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
    class_counts = pd.Series(labels).value_counts().reindex(range(num_classes), fill_value=0).astype(int)
    class_weights = pd.Series(0.0, index=range(num_classes), dtype=np.float32)

    nonzero = class_counts > 0
    class_weights.loc[nonzero] = 1.0 / class_counts.loc[nonzero].astype(np.float32)
    if nonzero.any():
        class_weights.loc[nonzero] = class_weights.loc[nonzero] / class_weights.loc[nonzero].mean()

    return class_counts, class_weights


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

    train_ds = ImageDataset(train_samples, build_transform(train=True))
    val_ds = ImageDataset(val_samples, build_transform(train=False))

    train_class_counts, class_weights = compute_class_stats(train_labels)
    val_class_counts, _ = compute_class_stats(val_labels)

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        common_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})

    train_loader = DataLoader(train_ds, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)

    return train_loader, val_loader, class_weights, train_class_counts, val_class_counts


# ===================== MODEL =====================
def build_model(device):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model.to(device)


# ===================== LOGGING =====================
def initialize_log_file(log_path, class_weights, train_class_counts, val_class_counts):
    with open(log_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING RUN LOG\n")
        f.write("=" * 80 + "\n")
        f.write(f"Run started            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model                  : {model_name}\n")
        f.write(f"Num classes            : {num_classes}\n")
        f.write(f"Batch size             : {batch_size}\n")
        f.write(f"Learning rate          : {learning_rate}\n")
        f.write(f"Num epochs             : {num_epochs}\n")
        f.write("Optimizer              : AdamW\n")
        f.write("Train loss             : Weighted CrossEntropyLoss + MixUp\n")
        f.write("Validation loss        : CrossEntropyLoss (logged) + Weighted CrossEntropyLoss (plotted)\n")
        f.write("Imbalance handling     : Weighted CrossEntropyLoss only\n")
        f.write(f"Top model saving       : Top {top_k_models} by validation QWK\n")
        f.write("Plots                  : Weighted CE loss curve + per-class recall curve\n")
        f.write(f"Seed                   : {SEED}\n\n")

        f.write("IMPORTANT TECHNIQUES USED\n")
        f.write("-" * 80 + "\n")
        f.write("Train augmentation     : RandomHorizontalFlip, ColorJitter(brightness=0.2, contrast=0.2), RandomAffine(degrees=0, translate=(0.05,0.05))\n")
        f.write("Loss function          : Weighted CrossEntropyLoss\n")
        f.write(f"MixUp                  : Enabled (alpha={mixup_alpha}, prob={mixup_prob})\n")
        f.write("Imbalance technique    : Class-weighted cross entropy\n")
        f.write("Metrics                : Train weighted CE loss, Validation CE loss, Validation weighted CE loss, QWK, Macro F1, per-class recall\n\n")

        f.write("CLASS STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train class counts     : {train_class_counts.to_dict()}\n")
        f.write(f"Val class counts       : {val_class_counts.to_dict()}\n")
        f.write(f"CE class weights       : {class_weights.round(6).to_dict()}\n\n")

        f.write("EPOCH METRICS\n")
        f.write("-" * 80 + "\n")


def append_resume_log(log_path, start_epoch):
    with open(log_path, "a") as f:
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write(f"Resumed training at    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Resuming from epoch    : {start_epoch}\n")
        f.write("-" * 80 + "\n")


def log_epoch(log_path, epoch, train_loss, metrics, lr):
    recall_str = ", ".join(f"class_{i}={r:.4f}" for i, r in enumerate(metrics["recall_per_class"]))
    with open(log_path, "a") as f:
        f.write(
            f"Epoch {epoch:02d} | "
            f"LR_used={lr:.8f} | "
            f"TrainWeightedCELoss={train_loss:.4f} | "
            f"ValCELoss={metrics['val_loss']:.4f} | "
            f"ValWeightedCELoss={metrics['val_weighted_loss']:.4f} | "
            f"QWK={metrics['qwk']:.4f} | "
            f"MacroF1={metrics['macro_f1']:.4f} | "
            f"Recall[{recall_str}]\n"
        )


def log_top_models_summary(log_path, top_models):
    with open(log_path, "a") as f:
        f.write("\n")
        f.write("TOP SAVED MODELS\n")
        f.write("-" * 80 + "\n")
        if not top_models:
            f.write("No top models were saved.\n")
            return
        for rank, item in enumerate(top_models, start=1):
            f.write(
                f"Rank {rank}: Epoch={item['epoch']}, "
                f"QWK={item['qwk']:.4f}, Path={item['path']}\n"
            )


# ===================== HISTORY / PLOTTING =====================
def history_columns():
    return ["epoch", "train_loss", "val_loss", "val_weighted_loss", "qwk", "macro_f1"] + [
        f"recall_class_{i}" for i in range(num_classes)
    ]


def empty_history_df():
    return pd.DataFrame(columns=history_columns())


def reset_history_artifacts():
    for path in (history_csv_path, loss_plot_path, recall_plot_path):
        if os.path.isfile(path):
            os.remove(path)


def load_history_csv_if_available():
    if not os.path.isfile(history_csv_path):
        return empty_history_df()

    df = pd.read_csv(history_csv_path)
    expected = set(history_columns())
    if not expected.issubset(df.columns):
        print("[Warning] Existing metrics_history.csv is missing columns. Starting a new history.")
        return empty_history_df()

    return df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)


def history_from_checkpoint(history_data):
    if not history_data:
        return empty_history_df()
    df = pd.DataFrame(history_data)
    expected = set(history_columns())
    if not expected.issubset(df.columns):
        return empty_history_df()
    return df.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)


def upsert_history_row(history_df, epoch, train_loss, metrics):
    row = {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(metrics["val_loss"]),
        "val_weighted_loss": float(metrics["val_weighted_loss"]),
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
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train Weighted CE Loss")
    plt.plot(history_df["epoch"], history_df["val_weighted_loss"], marker="o", label="Validation Weighted CE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Weighted CE Loss")
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
def train_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc="Training", leave=False)
    for x, y in loop:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        apply_mixup = (
            mixup_alpha > 0
            and mixup_prob > 0
            and x.size(0) > 1
            and random.random() < mixup_prob
        )

        if apply_mixup:
            mixed_x, y_a, y_b, lam = mixup_batch(x, y, alpha=mixup_alpha)
            with amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(mixed_x)
                loss = lam * criterion(out, y_a) + (1.0 - lam) * criterion(out, y_b)
        else:
            with amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(1, len(loader))


def evaluate(model, loader, ce_criterion, weighted_ce_criterion, device, use_amp):
    model.eval()
    total_ce_loss, total_weighted_ce_loss = 0.0, 0.0
    preds, labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                ce_loss = ce_criterion(out, y)
                weighted_ce_loss = weighted_ce_criterion(out, y)

            total_ce_loss += ce_loss.item()
            total_weighted_ce_loss += weighted_ce_loss.item()
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    qwk = cohen_kappa_score(labels, preds, labels=list(range(num_classes)), weights="quadratic")
    qwk = 0.0 if np.isnan(qwk) else qwk

    return {
        "val_loss": total_ce_loss / max(1, len(loader)),
        "val_weighted_loss": total_weighted_ce_loss / max(1, len(loader)),
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
        return top_models, None, None

    removed = None

    if len(top_models) >= top_k_models:
        removed = top_models.pop(-1)
        if os.path.isfile(removed["path"]):
            os.remove(removed["path"])

    save_path = os.path.join(models_dir, f"model_epoch_{epoch:03d}_qwk_{qwk:.4f}.pth")

    try:
        torch.save(model.state_dict(), save_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to save model to: {save_path}\n"
            f"This is often caused by insufficient disk space or user quota.\n"
            f"Original error: {e}"
        ) from e

    top_models.append({
        "epoch": epoch,
        "qwk": float(qwk),
        "path": save_path,
    })
    top_models = sort_top_models(top_models)

    return top_models, save_path, removed


def cleanup_stale_top_model_files(tracked_top_models):
    tracked_paths = {item["path"] for item in tracked_top_models}
    for path in glob.glob(os.path.join(models_dir, "*.pth")):
        if path not in tracked_paths and os.path.isfile(path):
            os.remove(path)


# ===================== RESUME CHECKPOINT =====================
def save_resume_checkpoint(epoch, model, optimizer, scheduler, scaler, last_metrics, class_weights, top_models, history_df):
    try:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "final_metrics": last_metrics,
                "class_weights": class_weights.to_dict(),
                "model_name": model_name,
                "num_classes": num_classes,
                "seed": SEED,
                "train_loss_type": "weighted_cross_entropy",
                "mixup_alpha": mixup_alpha,
                "mixup_prob": mixup_prob,
                "experiment_tag": EXPERIMENT_TAG,
                "top_models": top_models,
                "history": history_df.to_dict(orient="records"),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "torch_cuda_random_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            checkpoint_path,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to save checkpoint to: {checkpoint_path}\n"
            f"This is often caused by insufficient disk space or user quota.\n"
            f"Original error: {e}"
        ) from e


def load_checkpoint_if_available(model, optimizer, scheduler, scaler, device):
    if not os.path.isfile(checkpoint_path):
        return 1, None, [], empty_history_df()

    checkpoint = torch.load(checkpoint_path, map_location=device)

    ckpt_loss_type = checkpoint.get("train_loss_type", None)
    if ckpt_loss_type is not None and ckpt_loss_type != "weighted_cross_entropy":
        raise ValueError(
            f"Checkpoint train_loss_type={ckpt_loss_type} does not match current setup=weighted_cross_entropy. "
            f"Start a fresh experiment or use a matching checkpoint."
        )

    ckpt_mixup_alpha = checkpoint.get("mixup_alpha", None)
    ckpt_mixup_prob = checkpoint.get("mixup_prob", None)
    ckpt_experiment_tag = checkpoint.get("experiment_tag", None)

    if ckpt_experiment_tag is not None and ckpt_experiment_tag != EXPERIMENT_TAG:
        raise ValueError(
            f"Checkpoint experiment_tag={ckpt_experiment_tag} does not match current experiment_tag={EXPERIMENT_TAG}. "
            f"Start a fresh experiment or use the correct checkpoint."
        )

    if ckpt_mixup_alpha is not None and float(ckpt_mixup_alpha) != float(mixup_alpha):
        raise ValueError(
            f"Checkpoint mixup_alpha={ckpt_mixup_alpha} does not match current mixup_alpha={mixup_alpha}. "
            f"Use the same MixUp settings or start a fresh experiment."
        )

    if ckpt_mixup_prob is not None and float(ckpt_mixup_prob) != float(mixup_prob):
        raise ValueError(
            f"Checkpoint mixup_prob={ckpt_mixup_prob} does not match current mixup_prob={mixup_prob}. "
            f"Use the same MixUp settings or start a fresh experiment."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if "python_random_state" in checkpoint and checkpoint["python_random_state"] is not None:
        random.setstate(checkpoint["python_random_state"])
    if "numpy_random_state" in checkpoint and checkpoint["numpy_random_state"] is not None:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint and checkpoint["torch_random_state"] is not None:
        torch.set_rng_state(checkpoint["torch_random_state"])
    if (
        torch.cuda.is_available()
        and "torch_cuda_random_state_all" in checkpoint
        and checkpoint["torch_cuda_random_state_all"] is not None
    ):
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random_state_all"])

    last_epoch = int(checkpoint.get("epoch", 0))
    last_metrics = checkpoint.get("final_metrics", None)
    top_models = sort_top_models(checkpoint.get("top_models", []))

    history_df = history_from_checkpoint(checkpoint.get("history", []))
    if history_df.empty:
        history_df = load_history_csv_if_available()

    start_epoch = last_epoch + 1

    print(f"Found checkpoint: {checkpoint_path}")
    print(f"Resuming from epoch {start_epoch}")

    return start_epoch, last_metrics, top_models, history_df


# ===================== MAIN =====================
def main():
    check_startup_paths()
    check_disk_space(path=".", min_free_gb=2.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"Using device: {device}")
    print(f"Log file: {log_file}")

    label_dict = load_labels(excel_path)
    train_loader, val_loader, class_weights, train_class_counts, val_class_counts = get_loaders(label_dict, device)

    class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float32, device=device)
    train_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    val_ce_criterion = nn.CrossEntropyLoss()
    val_weighted_ce_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    model = build_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, last_metrics, top_models, history_df = load_checkpoint_if_available(
        model, optimizer, scheduler, scaler, device
    )

    if start_epoch == 1:
        initialize_log_file(log_file, class_weights, train_class_counts, val_class_counts)
        cleanup_stale_top_model_files([])
        reset_history_artifacts()
        top_models = []
        history_df = empty_history_df()
    else:
        append_resume_log(log_file, start_epoch)
        cleanup_stale_top_model_files(top_models)
        save_history_csv(history_df)
        plot_loss_curve(history_df)
        plot_recall_curve(history_df)

    if start_epoch > num_epochs:
        print(f"Checkpoint already corresponds to epoch {start_epoch - 1}, which is >= num_epochs={num_epochs}.")
        print("No further training was run.")
        return

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, train_criterion, device, scaler, use_amp)
        metrics = evaluate(model, val_loader, val_ce_criterion, val_weighted_ce_criterion, device, use_amp)
        last_metrics = metrics
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Weighted CE Loss={train_loss:.4f} | "
            f"Val CE Loss={metrics['val_loss']:.4f} | "
            f"Val Weighted CE Loss={metrics['val_weighted_loss']:.4f} | "
            f"QWK={metrics['qwk']:.4f} | "
            f"Macro F1={metrics['macro_f1']:.4f}"
        )
        print("Per-class Recall: " + ", ".join(
            f"class_{i}={r:.4f}" for i, r in enumerate(metrics["recall_per_class"])
        ))

        log_epoch(log_file, epoch, train_loss, metrics, current_lr)

        history_df = update_history_and_plots(history_df, epoch, train_loss, metrics)

        top_models, saved_path, removed_item = update_top_models(
            model=model,
            epoch=epoch,
            qwk=metrics["qwk"],
            top_models=top_models,
        )

        if saved_path is not None:
            print(f"Saved top model: {saved_path}")
        if removed_item is not None:
            print(f"Removed lower-ranked model: {removed_item['path']}")

        scheduler.step()

        save_resume_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            last_metrics=last_metrics,
            class_weights=class_weights,
            top_models=top_models,
            history_df=history_df,
        )

    with open(log_file, "a") as f:
        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write(f"Training completed at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if last_metrics is not None:
            f.write(f"Final Val CE Loss       : {last_metrics['val_loss']:.4f}\n")
            f.write(f"Final Val Weighted CE Loss: {last_metrics['val_weighted_loss']:.4f}\n")
            f.write(f"Final QWK               : {last_metrics['qwk']:.4f}\n")
            f.write(f"Final Macro F1          : {last_metrics['macro_f1']:.4f}\n")
            recall_str = ", ".join(f"class_{i}={r:.4f}" for i, r in enumerate(last_metrics["recall_per_class"]))
            f.write(f"Final Recall            : {recall_str}\n")
            f.write(f"Loss Plot               : {loss_plot_path}\n")
            f.write(f"Recall Plot             : {recall_plot_path}\n")

    log_top_models_summary(log_file, top_models)

    print("\nTraining complete.")
    print(f"Top {top_k_models} model files kept in: {models_dir}")
    print(f"Train split saved to: {train_split_path}")
    print(f"Val split saved to: {val_split_path}")
    print(f"Metrics history saved to: {history_csv_path}")
    print(f"Loss plot saved to: {loss_plot_path}")
    print(f"Recall plot saved to: {recall_plot_path}")
    print(f"Resume checkpoint saved to: {checkpoint_path}")
    if last_metrics is not None:
        print(f"Final QWK: {last_metrics['qwk']:.4f}")
    print(f"Metrics logged to: {log_file}")


if __name__ == "__main__":
    main()
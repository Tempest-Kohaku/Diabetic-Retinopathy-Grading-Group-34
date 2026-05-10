# ====================================================================== #
# python3 -m venv venv
# source venv/bin/activate
# pip install torch torchvision pandas scikit-learn tqdm timm numpy pillow matplotlib coral-pytorch safetensors
# ====================================================================== #

"""
Training script for diabetic retinopathy grading using a dual-backbone model.

Model architecture:
    EfficientNet-B4 branch + Swin Transformer branch + fusion head + CORN ordinal output.

Main features:
    - Loads retinal images and labels from a CSV file.
    - Creates or reuses a fixed train/validation split.
    - Applies training augmentations.
    - Trains using CORN loss for ordinal diabetic retinopathy grading.
    - Evaluates using Quadratic Weighted Kappa, Macro F1, and per-class recall.
    - Saves metrics and training plots after each epoch.
    - Keeps only the top-k best models based on validation QWK.
    - Optionally exports models as:
        1. Standard PyTorch .pth checkpoints
        2. Hugging Face-compatible custom inference folders
        3. Both formats
        4. No saved model files

Save format options:
    cfg.save_format = "pth"
        Saves only a PyTorch .pth checkpoint.

    cfg.save_format = "hf"
        Saves only Hugging Face-compatible files.

    cfg.save_format = "both"
        Saves both .pth and Hugging Face-compatible files.

    cfg.save_format = "none"
        Trains and validates the model but does not save model files.

Important:
    This model is intended for research purposes only and should not be used
    for clinical diagnosis.
"""

import os
import random
import logging
import shutil
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

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

try:
    from safetensors.torch import save_file as save_safetensors
except ImportError:
    save_safetensors = None


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    """
    Central configuration for the full training pipeline.

    This dataclass keeps all experiment settings in one place, which makes the
    code easier to reproduce, extend, and maintain.

    Main groups of settings:
        - Dataset paths
        - Model architecture names
        - Training hyperparameters
        - Augmentation parameters
        - Output directories
        - Model export options

    The save_format option controls how top-performing models are saved:

        "pth":
            Saves a standard PyTorch checkpoint.

        "hf":
            Saves a Hugging Face-compatible folder containing:
                - model.safetensors
                - config.json
                - architecture.py
                - handler.py
                - requirements.txt
                - README.md

        "both":
            Saves both .pth and Hugging Face formats.

        "none":
            Does not save model files, but still runs training and validation.
    """

    seed: int = 3
    experiment_tag: str = "exp_parallel_effb4_swinb384_corn_fusion_head"
    split_tag: str = "dr_fixed_split_v1"

    data_dir: Path = Path(r"/user/HS401/bs01338/Downloads/CLAHE/Train")
    csv_path: Path = Path(r"/user/HS401/bs01338/Downloads/DRG Dataset 384/train.csv")

    image_col: str = "image"
    label_col: str = "level"
    num_classes: int = 5

    batch_size: int = 14
    num_epochs: int = 20
    learning_rate: float = 3e-5
    weight_decay: float = 1e-4
    val_size: float = 0.17

    efficientnet_name: str = "efficientnet_b4.ra2_in1k"
    swin_name: str = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"
    pretrained: bool = True

    fusion_hidden_dim: int = 1024
    fusion_dropout: float = 0.3

    random_erasing_p: float = 0.25
    random_erasing_scale: tuple = (0.02, 0.10)
    random_erasing_ratio: tuple = (0.3, 3.3)
    random_erasing_value: str = "random"

    top_k_models: int = 2
    grad_clip: float = 1.0
    enable_grad_checkpointing: bool = True

    num_workers: int = min(4, os.cpu_count() or 1)

    output_dir: Path = Path("outputs")

    # ============================================================
    # MODEL SAVING / EXPORT OPTIONS
    # ============================================================
    # save_format controls which model artifacts are saved when a model
    # enters the top-k best validation results.
    #
    # "pth":
    #     Saves a normal PyTorch checkpoint.
    #
    # "hf":
    #     Saves a Hugging Face-compatible deployment folder.
    #
    # "both":
    #     Saves both .pth and Hugging Face files.
    #
    # "none":
    #     Does not save model files. Training, validation, metrics, and plots
    #     are still produced.
    # ============================================================
    save_format: str = "both"

    # If True, the .pth file stores everything needed to resume training:
    # model weights, optimizer, scheduler, AMP scaler, epoch, metrics, config.
    #
    # If False, the .pth file stores only model.state_dict().
    save_full_pth_checkpoint: bool = True

    image_size: int = 384

    id2label: dict = field(default_factory=lambda: {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR",
    })

    @property
    def run_dir(self):
        return self.output_dir / self.experiment_tag

    @property
    def logs_dir(self):
        return self.run_dir / "logs"

    @property
    def models_dir(self):
        return self.run_dir / "models"

    @property
    def plots_dir(self):
        return self.run_dir / "plots"

    @property
    def splits_dir(self):
        return self.run_dir / "splits"

    @property
    def log_path(self):
        return self.logs_dir / "train.log"

    @property
    def metrics_path(self):
        return self.logs_dir / "metrics_history.csv"

    @property
    def train_split_path(self):
        return self.splits_dir / f"train_split_{self.split_tag}.csv"

    @property
    def val_split_path(self):
        return self.splits_dir / f"val_split_{self.split_tag}.csv"

    @property
    def loss_plot_path(self):
        return self.plots_dir / "train_val_loss.png"

    @property
    def recall_plot_path(self):
        return self.plots_dir / "per_class_recall.png"


CFG = Config()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# SETUP
# ============================================================

def make_dirs(cfg: Config):
    """
    Creates the output directories required for logs, models, plots and splits.
    """

    for folder in [cfg.logs_dir, cfg.models_dir, cfg.plots_dir, cfg.splits_dir]:
        folder.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Config):
    """
    Sets up logging to both the terminal and the train.log file.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(cfg.log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def set_seed(seed: int):
    """
    Sets random seeds for Python, NumPy and PyTorch.

    This improves reproducibility by making data splitting, shuffling and model
    behaviour more consistent between runs.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    Sets a deterministic seed for each DataLoader worker.
    """

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def torch_generator(seed: int):
    """
    Creates a seeded PyTorch generator for deterministic DataLoader shuffling.
    """

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def check_paths(cfg: Config):
    """
    Checks that the required image directory and CSV file exist.
    """

    if not cfg.data_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {cfg.data_dir}")

    if not cfg.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {cfg.csv_path}")


def check_disk_space(path=".", min_free_gb=2.0):
    """
    Logs available disk space and warns if it is lower than the chosen threshold.
    """

    free_gb = shutil.disk_usage(path).free / (1024 ** 3)
    logging.info(f"Free disk space: {free_gb:.2f} GB")

    if free_gb < min_free_gb:
        logging.warning(f"Low disk space: less than {min_free_gb:.1f} GB free.")


def validate_config(cfg: Config):
    """
    Validates key configuration settings before training starts.

    This catches common mistakes early, such as invalid save formats, invalid
    class counts, invalid validation split size, or missing safetensors when
    Hugging Face export is requested.
    """

    valid_save_formats = {"none", "pth", "hf", "both"}

    if cfg.save_format not in valid_save_formats:
        raise ValueError(
            f"Invalid save_format='{cfg.save_format}'. "
            f"Choose from: {sorted(valid_save_formats)}"
        )

    if cfg.num_classes < 2:
        raise ValueError("num_classes must be at least 2 for CORN ordinal classification.")

    if cfg.top_k_models < 1:
        raise ValueError("top_k_models must be at least 1.")

    if not 0 < cfg.val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    if cfg.save_format in {"hf", "both"} and save_safetensors is None:
        raise ImportError(
            "safetensors is required for Hugging Face export. "
            "Install it with: pip install safetensors"
        )


# ============================================================
# TRANSFORMS
# ============================================================

def build_train_transform(cfg: Config):
    """
    Builds the transform pipeline used for training images.

    The training transform applies augmentation to improve generalisation:
        - Random horizontal flip
        - Colour jitter
        - Small translation
        - Normalisation
        - Random erasing

    Random Erasing is applied after normalisation because torchvision expects it
    to operate on tensors.
    """

    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(
            p=cfg.random_erasing_p,
            scale=cfg.random_erasing_scale,
            ratio=cfg.random_erasing_ratio,
            value=cfg.random_erasing_value,
            inplace=False,
        ),
    ])


def build_eval_transform():
    """
    Builds the transform pipeline used for validation images.

    Validation uses deterministic preprocessing only. No random augmentation is
    applied to validation data.
    """

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ============================================================
# DATASET
# ============================================================

class RetinalDataset(Dataset):
    """
    PyTorch Dataset for loading retinal fundus images.

    Each sample contains:
        - image path
        - integer diabetic retinopathy label

    The dataset:
        - opens each image using PIL
        - converts it to RGB
        - applies the provided torchvision transform
        - returns the transformed image tensor and label

    This dataset is used for both training and validation. The difference
    between training and validation behaviour comes from the transform passed
    into the dataset.
    """

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {path}") from e

        image = self.transform(image)
        return image, int(label)


# ============================================================
# DATA LOADING
# ============================================================

def load_labels(cfg: Config):
    """
    Loads labels from the CSV file and validates required columns.

    Expected CSV columns:
        cfg.image_col:
            image filename or image ID

        cfg.label_col:
            integer diabetic retinopathy grade from 0 to num_classes - 1
    """

    df = pd.read_csv(cfg.csv_path)

    required_cols = {cfg.image_col, cfg.label_col}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df[cfg.image_col] = df[cfg.image_col].astype(str).str.strip()
    df[cfg.label_col] = df[cfg.label_col].astype(int)

    valid_labels = set(range(cfg.num_classes))
    found_labels = set(df[cfg.label_col].unique())

    if not found_labels.issubset(valid_labels):
        raise ValueError(
            f"Found labels outside 0-{cfg.num_classes - 1}: {sorted(found_labels)}"
        )

    duplicates = df[df[cfg.image_col].duplicated()][cfg.image_col].tolist()
    if duplicates:
        raise ValueError(f"Duplicate image names in CSV. Examples: {duplicates[:10]}")

    return df


def build_image_index(cfg: Config):
    """
    Builds a dictionary mapping image basenames to full file paths.

    This allows the CSV to contain image names with or without extensions.
    Example:
        CSV image name: 12345
        File on disk:   12345.jpg
    """

    image_index = {}
    duplicates = []

    for path in cfg.data_dir.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        key = path.stem.strip().lower()

        if key in image_index:
            duplicates.append(key)

        image_index[key] = path

    if duplicates:
        raise ValueError(f"Duplicate image basenames found. Examples: {duplicates[:10]}")

    return image_index


def build_samples(cfg: Config):
    """
    Matches CSV entries to image files on disk.

    Returns:
        A list of tuples:
            [(image_path, label), ...]
    """

    df = load_labels(cfg)
    image_index = build_image_index(cfg)

    samples = []
    missing = []

    for _, row in df.iterrows():
        image_name = str(row[cfg.image_col]).strip()
        label = int(row[cfg.label_col])

        key = Path(image_name).stem.strip().lower()

        if key in image_index:
            samples.append((image_index[key], label))
        else:
            missing.append(image_name)

    logging.info("\n================ FILE MATCHING ================")
    logging.info(f"CSV entries        : {len(df)}")
    logging.info(f"Matched images     : {len(samples)}")
    logging.info(f"Missing images     : {len(missing)}")

    if missing:
        logging.info(f"Example missing    : {missing[:20]}")

    logging.info("==============================================\n")

    if not samples:
        raise ValueError("No matched image files found.")

    return samples


def class_counts(samples, cfg: Config):
    """
    Counts the number of samples per class.
    """

    labels = [label for _, label in samples]
    return pd.Series(labels).value_counts().reindex(
        range(cfg.num_classes), fill_value=0
    ).astype(int)


def save_split(samples, path):
    """
    Saves a train or validation split to CSV.
    """

    pd.DataFrame(samples, columns=["path", "label"]).to_csv(path, index=False)


def load_split(path):
    """
    Loads a saved train or validation split from CSV.
    """

    df = pd.read_csv(path)

    if not {"path", "label"}.issubset(df.columns):
        raise ValueError(f"Split file must contain path and label columns: {path}")

    return list(zip(df["path"].map(Path), df["label"].astype(int)))


def get_train_val_samples(cfg: Config):
    """
    Returns train and validation samples.

    If split files already exist, they are reused. This keeps the same split
    across multiple runs, which makes experiments more comparable.

    If split files do not exist, a new stratified split is created and saved.
    """

    if cfg.train_split_path.exists() and cfg.val_split_path.exists():
        logging.info("Using existing train/validation split files.")
        train_samples = load_split(cfg.train_split_path)
        val_samples = load_split(cfg.val_split_path)
        return train_samples, val_samples

    samples = build_samples(cfg)

    paths = [path for path, _ in samples]
    labels = [label for _, label in samples]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=cfg.val_size,
        stratify=labels,
        random_state=cfg.seed,
    )

    train_samples = list(zip(train_paths, train_labels))
    val_samples = list(zip(val_paths, val_labels))

    save_split(train_samples, cfg.train_split_path)
    save_split(val_samples, cfg.val_split_path)

    logging.info("Created and saved new train/validation split files.")

    return train_samples, val_samples


def build_loaders(cfg: Config, device):
    """
    Builds PyTorch DataLoaders for training and validation.

    Efficiency settings:
        - pin_memory is enabled when using CUDA.
        - persistent_workers is enabled when num_workers > 0.
        - prefetch_factor allows workers to prepare batches in advance.
    """

    train_samples, val_samples = get_train_val_samples(cfg)

    train_dataset = RetinalDataset(
        train_samples,
        transform=build_train_transform(cfg),
    )

    val_dataset = RetinalDataset(
        val_samples,
        transform=build_eval_transform(),
    )

    common_args = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }

    if cfg.num_workers > 0:
        common_args["persistent_workers"] = True
        common_args["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch_generator(cfg.seed),
        **common_args,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_args,
    )

    return train_loader, val_loader, train_samples, val_samples


# ============================================================
# MODEL
# ============================================================

class ParallelEfficientNetSwin(nn.Module):
    """
    Dual-backbone diabetic retinopathy grading model.

    Architecture:
        1. EfficientNet backbone extracts convolutional image features.
        2. Swin Transformer backbone extracts transformer-based visual features.
        3. Features from both branches are converted to vectors.
        4. The vectors are concatenated.
        5. A fusion head predicts CORN ordinal logits.

    CORN output:
        For num_classes = 5, the model outputs 4 logits.

        These logits represent ordinal thresholds rather than standard
        5-class softmax probabilities.

    Important:
        The module names in this class must remain consistent when loading
        saved checkpoints. For example:
            - efficientnet
            - swin
            - fusion_head

        Changing these names will cause load_state_dict(strict=True) to fail
        unless the checkpoint keys are also updated.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.efficientnet = timm.create_model(
            cfg.efficientnet_name,
            pretrained=cfg.pretrained,
            num_classes=0,
        )

        self.swin = timm.create_model(
            cfg.swin_name,
            pretrained=cfg.pretrained,
            num_classes=0,
        )

        eff_dim = getattr(self.efficientnet, "num_features", None)
        swin_dim = getattr(self.swin, "num_features", None)

        if eff_dim is None or swin_dim is None:
            raise ValueError("Could not determine backbone feature dimensions.")

        fusion_dim = eff_dim + swin_dim

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(cfg.fusion_hidden_dim, cfg.num_classes - 1),
        )

    @staticmethod
    def to_vector(features):
        """
        Converts backbone outputs into 2D feature vectors.

        Supports common output shapes:
            - [batch, features]
            - [batch, tokens, features]
            - [batch, channels, height, width]
            - [batch, height, width, channels]
        """

        if features.ndim == 2:
            return features

        if features.ndim == 3:
            return features.mean(dim=1)

        if features.ndim == 4:
            # Handles both NCHW and NHWC feature maps.
            if (
                features.shape[1] > features.shape[-1]
                and features.shape[1] > features.shape[-2]
            ):
                return features.mean(dim=(2, 3))

            return features.mean(dim=(1, 2))

        raise ValueError(f"Unexpected feature shape: {features.shape}")

    def forward(self, x):
        eff_features = self.to_vector(self.efficientnet(x))
        swin_features = self.to_vector(self.swin(x))

        fused = torch.cat([eff_features, swin_features], dim=1)
        return self.fusion_head(fused)


def build_model(cfg: Config, device):
    """
    Builds the dual-backbone model and moves it to the selected device.

    Gradient checkpointing is enabled when supported by the backbone models.
    This can reduce GPU memory usage during training, although it may slightly
    increase computation time.

    Args:
        cfg:
            Experiment configuration.
        device:
            CUDA or CPU device.

    Returns:
        The initialized PyTorch model.
    """

    model = ParallelEfficientNetSwin(cfg)

    if cfg.enable_grad_checkpointing:
        for backbone in [model.efficientnet, model.swin]:
            if hasattr(backbone, "set_grad_checkpointing"):
                backbone.set_grad_checkpointing(True)

    return model.to(device)


# ============================================================
# METRICS AND PLOTS
# ============================================================

def calculate_metrics(labels, preds, val_loss, cfg: Config):
    """
    Calculates validation metrics.

    Metrics:
        - validation CORN loss
        - Quadratic Weighted Kappa
        - Macro F1
        - per-class recall
    """

    qwk = cohen_kappa_score(
        labels,
        preds,
        labels=list(range(cfg.num_classes)),
        weights="quadratic",
    )

    if np.isnan(qwk):
        qwk = 0.0

    return {
        "val_loss": float(val_loss),
        "qwk": float(qwk),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "recall_per_class": recall_score(
            labels,
            preds,
            average=None,
            labels=list(range(cfg.num_classes)),
            zero_division=0,
        ),
    }


def save_metrics_csv(history, cfg: Config):
    """
    Saves epoch-by-epoch training and validation metrics to CSV.
    """

    rows = []

    for item in history:
        row = {
            "epoch": item["epoch"],
            "lr": item["lr"],
            "train_loss": item["train_loss"],
            "val_loss": item["val_loss"],
            "qwk": item["qwk"],
            "macro_f1": item["macro_f1"],
        }

        for i, recall_value in enumerate(item["recall_per_class"]):
            row[f"recall_class_{i}"] = recall_value

        rows.append(row)

    pd.DataFrame(rows).to_csv(cfg.metrics_path, index=False)


def plot_training_curves(history, cfg: Config):
    """
    Saves training plots:
        - training vs validation loss
        - per-class recall across epochs
    """

    if not history:
        return

    df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train CORN Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation CORN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.loss_plot_path, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))

    recall_values = np.stack(df["recall_per_class"].values)

    for class_idx in range(cfg.num_classes):
        plt.plot(
            df["epoch"],
            recall_values[:, class_idx],
            marker="o",
            label=f"Class {class_idx}",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Per-Class Recall vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.recall_plot_path, dpi=200)
    plt.close()


# ============================================================
# EXPORT HELPERS
# ============================================================

def to_jsonable(value):
    """
    Converts Python objects into JSON-safe values.

    Handles:
        - pathlib.Path
        - tuples
        - lists
        - dictionaries
        - NumPy numbers
        - NumPy arrays
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]

    if isinstance(value, list):
        return [to_jsonable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


def build_export_config(cfg: Config, epoch: int, metrics: Dict[str, Any]):
    """
    Builds a JSON-compatible configuration dictionary for saved models.

    This config is stored in:
        - .pth checkpoints
        - Hugging Face config.json files

    It includes:
        - model architecture details
        - class labels
        - image preprocessing settings
        - validation metrics
        - original training configuration

    This makes the saved model easier to reproduce and deploy later.
    """

    cfg_dict = {
        key: to_jsonable(value)
        for key, value in asdict(cfg).items()
    }

    id2label = {str(k): v for k, v in cfg.id2label.items()}
    label2id = {v: int(k) for k, v in cfg.id2label.items()}

    return {
        "architecture": "ParallelEfficientNetSwin",
        "task": "diabetic_retinopathy_grading",
        "loss_type": "CORN",
        "num_classes": cfg.num_classes,
        "num_corn_outputs": cfg.num_classes - 1,
        "efficientnet_name": cfg.efficientnet_name,
        "swin_name": cfg.swin_name,
        "pretrained": False,
        "fusion_hidden_dim": cfg.fusion_hidden_dim,
        "fusion_dropout": cfg.fusion_dropout,
        "image_size": cfg.image_size,
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "id2label": id2label,
        "label2id": label2id,
        "epoch": int(epoch),
        "metrics": {
            key: to_jsonable(value)
            for key, value in metrics.items()
        },
        "training_config": cfg_dict,
    }


def atomic_torch_save(obj, save_path: Path):
    """
    Saves a PyTorch file safely.

    The file is first written to a temporary path and then renamed. This reduces
    the risk of leaving a corrupted checkpoint if saving is interrupted.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = save_path.with_suffix(save_path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(save_path)


def write_text(path: Path, content: str):
    """
    Writes text content to a file using UTF-8 encoding.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def save_pth_artifact(
    model: nn.Module,
    save_path: Path,
    cfg: Config,
    epoch: int,
    metrics: Dict[str, Any],
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
):
    """
    Saves the model in PyTorch .pth format.

    When cfg.save_full_pth_checkpoint is True, the file contains:
        - model_state_dict
        - optimizer_state_dict
        - scheduler_state_dict
        - scaler_state_dict
        - epoch
        - metrics
        - export config

    This is useful for resuming training or performing local testing.

    When cfg.save_full_pth_checkpoint is False, only model.state_dict() is saved.
    This creates a smaller file but contains less training information.
    """

    if cfg.save_full_pth_checkpoint:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "metrics": {
                key: to_jsonable(value)
                for key, value in metrics.items()
            },
            "config": build_export_config(cfg, epoch, metrics),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()

        atomic_torch_save(checkpoint, save_path)

    else:
        atomic_torch_save(model.state_dict(), save_path)


def save_huggingface_artifact(
    model: nn.Module,
    save_dir: Path,
    cfg: Config,
    epoch: int,
    metrics: Dict[str, Any],
):
    """
    Saves a Hugging Face-compatible custom model folder.

    The exported folder contains:
        - model.safetensors:
            model weights in the safer Hugging Face tensor format

        - config.json:
            architecture, preprocessing, labels, and metric information

        - architecture.py:
            model class needed to reconstruct the PyTorch architecture

        - handler.py:
            custom inference handler for processing images and returning results

        - requirements.txt:
            Python dependencies required by the model

        - README.md:
            model card with architecture details and research-use disclaimer

    This format is intended for deploying the model on Hugging Face using a
    custom inference handler.
    """

    if save_safetensors is None:
        raise ImportError(
            "safetensors is required for Hugging Face export. "
            "Install it with: pip install safetensors"
        )

    tmp_dir = save_dir.with_name(save_dir.name + "_tmp")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    if save_dir.exists():
        shutil.rmtree(save_dir)

    tmp_dir.mkdir(parents=True, exist_ok=True)

    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    save_safetensors(cpu_state_dict, tmp_dir / "model.safetensors")

    export_config = build_export_config(cfg, epoch, metrics)

    with open(tmp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(export_config, f, indent=2)

    write_text(tmp_dir / "architecture.py", get_hf_architecture_py())
    write_text(tmp_dir / "handler.py", get_hf_handler_py())
    write_text(tmp_dir / "requirements.txt", get_hf_requirements_txt())
    write_text(tmp_dir / "README.md", get_hf_readme_md(cfg, epoch, metrics))

    tmp_dir.replace(save_dir)


def remove_artifact(path: Path):
    """
    Removes a saved artifact, either a file or directory.
    """

    if path.is_file():
        path.unlink()

    elif path.is_dir():
        shutil.rmtree(path)


def get_hf_architecture_py():
    """
    Returns the Python source code for the Hugging Face architecture.py file.

    This architecture must match the training model exactly so that
    model.safetensors can be loaded correctly.
    """

    return r'''
import torch
import torch.nn as nn
import timm


class ParallelEfficientNetSwin(nn.Module):
    """
    Model architecture used for Hugging Face inference.

    This class must match the architecture used during training exactly.
    The layer names must also match the saved model weights.

    The model combines:
        - EfficientNet features
        - Swin Transformer features
        - Fusion head
        - CORN ordinal output layer
    """

    def __init__(
        self,
        efficientnet_name,
        swin_name,
        num_classes=5,
        fusion_hidden_dim=1024,
        fusion_dropout=0.3,
    ):
        super().__init__()

        self.efficientnet = timm.create_model(
            efficientnet_name,
            pretrained=False,
            num_classes=0,
        )

        self.swin = timm.create_model(
            swin_name,
            pretrained=False,
            num_classes=0,
        )

        eff_dim = getattr(self.efficientnet, "num_features", None)
        swin_dim = getattr(self.swin, "num_features", None)

        if eff_dim is None or swin_dim is None:
            raise ValueError("Could not determine backbone feature dimensions.")

        fusion_dim = eff_dim + swin_dim

        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, num_classes - 1),
        )

    @staticmethod
    def to_vector(features):
        if features.ndim == 2:
            return features

        if features.ndim == 3:
            return features.mean(dim=1)

        if features.ndim == 4:
            if (
                features.shape[1] > features.shape[-1]
                and features.shape[1] > features.shape[-2]
            ):
                return features.mean(dim=(2, 3))

            return features.mean(dim=(1, 2))

        raise ValueError(f"Unexpected feature shape: {features.shape}")

    def forward(self, x):
        eff_features = self.to_vector(self.efficientnet(x))
        swin_features = self.to_vector(self.swin(x))

        fused = torch.cat([eff_features, swin_features], dim=1)
        return self.fusion_head(fused)
'''


def get_hf_handler_py():
    """
    Returns the Python source code for the Hugging Face handler.py file.

    The handler is responsible for loading the model, preprocessing an input
    image, running inference, and returning JSON output.
    """

    return r'''
import base64
import io
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from architecture import ParallelEfficientNetSwin


class EndpointHandler:
    """
    Custom Hugging Face inference handler.

    This class is loaded by Hugging Face when the model is deployed.

    It performs:
        1. model loading
        2. image decoding
        3. preprocessing
        4. model inference
        5. CORN ordinal postprocessing
        6. JSON response formatting

    The output contains:
        - predicted_class
        - label
        - corn_probabilities
        - raw logits
    """

    def __init__(self, path=""):
        self.model_dir = Path(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.model_dir / "config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model = ParallelEfficientNetSwin(
            efficientnet_name=self.config["efficientnet_name"],
            swin_name=self.config["swin_name"],
            num_classes=self.config["num_classes"],
            fusion_hidden_dim=self.config["fusion_hidden_dim"],
            fusion_dropout=self.config["fusion_dropout"],
        )

        state_dict = load_file(str(self.model_dir / "model.safetensors"))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        image_size = int(self.config["image_size"])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config["normalization_mean"],
                std=self.config["normalization_std"],
            ),
        ])

    def decode_image(self, data):
        """
        Decodes an input image.

        Supported input formats:
            - raw image bytes
            - {"inputs": raw_bytes}
            - {"inputs": base64_string}
        """

        if isinstance(data, bytes):
            image_bytes = data

        elif isinstance(data, dict):
            inputs = data.get("inputs")

            if isinstance(inputs, bytes):
                image_bytes = inputs

            elif isinstance(inputs, str):
                if "," in inputs:
                    inputs = inputs.split(",", 1)[1]

                image_bytes = base64.b64decode(inputs)

            else:
                raise ValueError("Expected inputs to be bytes or a base64 string.")

        else:
            raise ValueError("Expected input data to be bytes or a dictionary.")

        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def corn_prediction_from_logits(logits, threshold=0.5):
        """
        Converts CORN logits into an ordinal class prediction.

        For five classes, the model outputs four logits. Each logit represents
        whether the image passes an ordinal threshold. The number of passed
        thresholds becomes the final predicted class.
        """

        probabilities = torch.sigmoid(logits)
        prediction = (probabilities > threshold).sum(dim=1)
        return prediction, probabilities

    def __call__(self, data):
        image = self.decode_image(data)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(image_tensor)
            prediction, probabilities = self.corn_prediction_from_logits(logits)

        pred_id = int(prediction.item())
        label = self.config["id2label"][str(pred_id)]

        return {
            "predicted_class": pred_id,
            "label": label,
            "corn_probabilities": probabilities.squeeze(0).cpu().tolist(),
            "logits": logits.squeeze(0).cpu().tolist(),
        }
'''


def get_hf_requirements_txt():
    """
    Returns dependencies required by the Hugging Face custom model.
    """

    return r'''
torch
torchvision
timm
safetensors
pillow
numpy
'''


def get_hf_readme_md(cfg: Config, epoch: int, metrics: Dict[str, Any]):
    """
    Builds a simple Hugging Face model card.
    """

    qwk = metrics.get("qwk", None)
    qwk_text = f"{qwk:.4f}" if qwk is not None else "N/A"

    return f'''
---
library_name: pytorch
pipeline_tag: image-classification
tags:
- diabetic-retinopathy
- image-classification
- pytorch
- timm
- corn
---

# {cfg.experiment_tag}

Custom PyTorch model for diabetic retinopathy grading.

## Architecture

- EfficientNet branch: `{cfg.efficientnet_name}`
- Swin Transformer branch: `{cfg.swin_name}`
- Fusion head: LayerNorm → Linear → GELU → Dropout → CORN output
- Number of DR classes: `{cfg.num_classes}`
- Number of CORN outputs: `{cfg.num_classes - 1}`

## Exported checkpoint

- Epoch: `{epoch}`
- Validation QWK: `{qwk_text}`

## Labels

| Class ID | Label |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## Important note

This model is for research purposes only and should not be used for clinical diagnosis.
'''


# ============================================================
# CHECKPOINT MANAGER
# ============================================================

class TopKCheckpointManager:
    """
    Manages saving only the top-k best models during training.

    Models are ranked using validation Quadratic Weighted Kappa, known as QWK.
    This is useful for diabetic retinopathy grading because the task is ordinal:
    predicting class 4 instead of class 3 is less severe than predicting class 4
    instead of class 0.

    The manager saves a new model only when:
        - fewer than top_k_models have been saved, or
        - the current model performs better than the worst saved top-k model.

    Depending on cfg.save_format, each selected model is saved as:
        - .pth checkpoint
        - Hugging Face-compatible folder
        - both
        - no artifact

    Older weaker checkpoints are automatically removed to save disk space.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.best_models = []

    def save_if_best(
        self,
        model,
        epoch,
        metrics,
        optimizer=None,
        scheduler=None,
        scaler=None,
    ):
        """
        Saves the model if it qualifies for the current top-k list.

        Args:
            model:
                Trained model after the current epoch.
            epoch:
                Current epoch number.
            metrics:
                Validation metrics from the current epoch.
            optimizer:
                Optimizer state, used only for full .pth checkpoint saving.
            scheduler:
                Scheduler state, used only for full .pth checkpoint saving.
            scaler:
                AMP scaler state, used only for full .pth checkpoint saving.

        Returns:
            List of saved artifact paths, or None if the model did not qualify.
        """

        qwk = float(metrics["qwk"])

        qualifies = (
            len(self.best_models) < self.cfg.top_k_models
            or qwk > min(self.best_models, key=lambda x: x["qwk"])["qwk"]
        )

        if not qualifies:
            return None

        artifact_base_name = f"model_epoch_{epoch:03d}_qwk_{qwk:.4f}"
        saved_artifacts = []

        if self.cfg.save_format in {"pth", "both"}:
            pth_path = self.cfg.models_dir / f"{artifact_base_name}.pth"

            save_pth_artifact(
                model=model,
                save_path=pth_path,
                cfg=self.cfg,
                epoch=epoch,
                metrics=metrics,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
            )

            saved_artifacts.append(pth_path)

        if self.cfg.save_format in {"hf", "both"}:
            hf_dir = self.cfg.models_dir / f"{artifact_base_name}_hf"

            save_huggingface_artifact(
                model=model,
                save_dir=hf_dir,
                cfg=self.cfg,
                epoch=epoch,
                metrics=metrics,
            )

            saved_artifacts.append(hf_dir)

        self.best_models.append({
            "epoch": int(epoch),
            "qwk": qwk,
            "artifacts": saved_artifacts,
        })

        self.best_models = sorted(
            self.best_models,
            key=lambda x: (x["qwk"], x["epoch"]),
            reverse=True,
        )

        while len(self.best_models) > self.cfg.top_k_models:
            removed = self.best_models.pop(-1)

            for artifact_path in removed["artifacts"]:
                if artifact_path.exists():
                    remove_artifact(artifact_path)

        return saved_artifacts


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    """
    Coordinates the full training and validation process.

    Responsibilities:
        - prepares data loaders
        - builds the model
        - sets up optimizer, scheduler, and mixed precision scaler
        - trains the model for each epoch
        - validates after each epoch
        - logs metrics
        - saves training curves
        - saves top-k model checkpoints using TopKCheckpointManager

    The Trainer class keeps the main training loop clean and separates
    training logic from configuration, dataset loading, model definition,
    metrics, and checkpoint export.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

        self.train_loader, self.val_loader, self.train_samples, self.val_samples = build_loaders(
            cfg,
            self.device,
        )

        self.model = build_model(cfg, self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.num_epochs,
        )

        self.scaler = amp.GradScaler(self.device.type, enabled=self.use_amp)
        self.checkpoints = TopKCheckpointManager(cfg)
        self.history = []

    def train_one_epoch(self, epoch):
        """
        Runs one full training epoch.

        For each batch:
            - moves images and labels to the selected device
            - performs forward pass
            - calculates CORN loss
            - performs backpropagation
            - clips gradients
            - updates model weights
            - updates the mixed precision scaler when CUDA is used

        Args:
            epoch:
                Current epoch number, used for progress display.

        Returns:
            Average training loss for the epoch.
        """

        self.model.train()
        total_loss = 0.0

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch} Training", leave=False)

        for images, labels in loop:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = corn_loss(outputs.float(), labels, num_classes=self.cfg.num_classes)

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(1, len(self.train_loader))

    def validate(self):
        """
        Evaluates the model on the validation set.

        Validation is performed without gradient calculation to reduce memory
        usage and improve speed.

        Metrics calculated:
            - validation CORN loss
            - Quadratic Weighted Kappa
            - Macro F1
            - per-class recall

        Returns:
            Dictionary containing validation metrics.
        """

        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = corn_loss(outputs.float(), labels, num_classes=self.cfg.num_classes)

                preds = corn_label_from_logits(outputs.float())

                total_loss += loss.item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(1, len(self.val_loader))

        return calculate_metrics(
            labels=all_labels,
            preds=all_preds,
            val_loss=avg_loss,
            cfg=self.cfg,
        )

    def log_start(self):
        """
        Logs experiment configuration and dataset summary before training.
        """

        train_counts = class_counts(self.train_samples, self.cfg)
        val_counts = class_counts(self.val_samples, self.cfg)

        logging.info("=" * 80)
        logging.info("TRAINING RUN")
        logging.info("=" * 80)
        logging.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info(f"Device: {self.device}")
        logging.info(f"Experiment: {self.cfg.experiment_tag}")
        logging.info(f"Save format: {self.cfg.save_format}")
        logging.info(f"Train samples: {len(self.train_samples)}")
        logging.info(f"Val samples: {len(self.val_samples)}")
        logging.info(f"Train class counts: {train_counts.to_dict()}")
        logging.info(f"Val class counts: {val_counts.to_dict()}")
        logging.info("-" * 80)

        logging.info("CONFIG")
        for key, value in asdict(self.cfg).items():
            logging.info(f"{key}: {value}")

        logging.info("=" * 80)

    def fit(self):
        """
        Runs the complete training process.

        For each epoch:
            1. Train for one epoch.
            2. Validate the model.
            3. Store metrics in memory.
            4. Save metrics to CSV.
            5. Update training plots.
            6. Save model if it enters the top-k best models.
            7. Step the learning-rate scheduler.

        At the end, the method logs the best validation QWK and the paths of
        all saved top model artifacts.
        """

        self.log_start()

        best_qwk = -1.0

        for epoch in range(1, self.cfg.num_epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]

            logging.info(f"\nEpoch {epoch}/{self.cfg.num_epochs}")
            logging.info(f"Learning rate: {current_lr:.8f}")

            train_loss = self.train_one_epoch(epoch)
            metrics = self.validate()

            epoch_result = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": float(train_loss),
                **metrics,
            }

            self.history.append(epoch_result)

            recall_str = ", ".join(
                f"class_{i}={r:.4f}"
                for i, r in enumerate(metrics["recall_per_class"])
            )

            logging.info(
                f"Train Loss={train_loss:.4f} | "
                f"Val Loss={metrics['val_loss']:.4f} | "
                f"QWK={metrics['qwk']:.4f} | "
                f"Macro F1={metrics['macro_f1']:.4f}"
            )

            logging.info(f"Recall: {recall_str}")

            saved_artifacts = self.checkpoints.save_if_best(
                model=self.model,
                epoch=epoch,
                metrics=metrics,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )

            if saved_artifacts:
                for artifact in saved_artifacts:
                    logging.info(f"Saved top model artifact: {artifact}")

            if metrics["qwk"] > best_qwk:
                best_qwk = metrics["qwk"]

            save_metrics_csv(self.history, self.cfg)
            plot_training_curves(self.history, self.cfg)

            self.scheduler.step()

        logging.info("\nTraining complete.")
        logging.info(f"Best validation QWK: {best_qwk:.4f}")
        logging.info(f"Metrics saved to: {self.cfg.metrics_path}")
        logging.info(f"Loss plot saved to: {self.cfg.loss_plot_path}")
        logging.info(f"Recall plot saved to: {self.cfg.recall_plot_path}")

        logging.info("\nTop saved models:")
        for rank, item in enumerate(self.checkpoints.best_models, start=1):
            artifact_text = ", ".join(str(path) for path in item["artifacts"])

            if not artifact_text:
                artifact_text = "No artifacts saved"

            logging.info(
                f"Rank {rank}: Epoch={item['epoch']}, "
                f"QWK={item['qwk']:.4f}, "
                f"Artifacts={artifact_text}"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main entry point for the training script.
    """

    make_dirs(CFG)
    setup_logging(CFG)
    validate_config(CFG)
    set_seed(CFG.seed)
    check_paths(CFG)
    check_disk_space(".", min_free_gb=2.0)

    trainer = Trainer(CFG)
    trainer.fit()


if __name__ == "__main__":
    main()
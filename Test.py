# ====================================================================== #
# python3 -m venv venv
# source venv/bin/activate
# pip install torch torchvision matplotlib pandas scikit-learn tqdm timm numpy pillow coral-pytorch
# ====================================================================== #

"""
Test/evaluation script for diabetic retinopathy grading.

Model architecture:
    EfficientNet-B4 branch + Swin Transformer branch + fusion head + CORN ordinal output.

This script:
    - Loads a trained PyTorch checkpoint.
    - Loads test images and labels from a CSV file.
    - Matches CSV image names to image files on disk.
    - Runs inference on the test set.
    - Calculates test metrics:
        - CORN loss
        - Accuracy
        - Quadratic Weighted Kappa
        - Micro F1
        - Weighted F1
        - Macro F1
        - Per-class F1
        - Per-class recall
    - Saves:
        - test_metrics.csv
        - test_predictions.csv
        - confusion_matrix.png
        - test.log

Important:
    The model class in this file must match the training model architecture.
 
This model is intended for research purposes only and should not be used for
clinical diagnosis.
"""

import os
import re
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score,
    cohen_kappa_score,
)

from tqdm import tqdm
import timm

from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    """
    Central configuration for the test/evaluation pipeline.

    This keeps paths, model settings, preprocessing settings, and output paths
    in one place, making the script easier to maintain and reuse.
    """

    experiment_name: str = "exp_parallel_effb4_swinb384_corn_fusion_head_test"

    model_path: Path = Path("/user/HS401/bs01338/Downloads/DRG/Diabetic-Retinopathy-Grading-Group-34/outputs/"
                            "exp_parallel_effb4_swinb384_corn_fusion_head/models/model_epoch_015_qwk_0.8122.pth"
    )

    test_dir: Path = Path("/user/HS401/bs01338/Downloads/CLAHE/Test")
    csv_path: Path = Path("/user/HS401/bs01338/Downloads/DRG Dataset 384/test.csv")

    image_col_options: tuple = ("image", "filename")
    label_col_options: tuple = ("level", "class")

    batch_size: int = 32
    num_classes: int = 5
    num_workers: int = min(4, os.cpu_count() or 1)

    image_size: int = 384

    efficientnet_name: str = "efficientnet_b4.ra2_in1k"
    swin_name: str = "swin_base_patch4_window12_384.ms_in22k_ft_in1k"

    fusion_hidden_dim: int = 1024
    fusion_dropout: float = 0.3

    output_dir: Path = Path("test_outputs")

    @property
    def run_dir(self):
        return self.output_dir / self.experiment_name

    @property
    def log_path(self):
        return self.run_dir / "test.log"

    @property
    def confusion_matrix_path(self):
        return self.run_dir / "confusion_matrix.png"

    @property
    def metrics_csv_path(self):
        return self.run_dir / "test_metrics.csv"

    @property
    def predictions_csv_path(self):
        return self.run_dir / "test_predictions.csv"


CFG = Config()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# SETUP
# ============================================================

def make_dirs(cfg: Config):
    """
    Creates the test output directory.
    """

    cfg.run_dir.mkdir(parents=True, exist_ok=True)


def setup_logging(cfg: Config):
    """
    Sets up logging to both terminal and test.log.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(cfg.log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )


def validate_config(cfg: Config):
    """
    Validates important configuration values before testing starts.
    """

    if cfg.num_classes < 2:
        raise ValueError("num_classes must be at least 2 for CORN ordinal classification.")

    if cfg.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if cfg.num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    if cfg.image_size < 1:
        raise ValueError("image_size must be at least 1.")


def check_paths(cfg: Config):
    """
    Checks that the checkpoint, test image folder, and CSV file exist.
    """

    if not cfg.model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {cfg.model_path}")

    if not cfg.test_dir.is_dir():
        raise FileNotFoundError(f"Test image folder not found: {cfg.test_dir}")

    if not cfg.csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {cfg.csv_path}")


def parse_epoch_from_path(path: Path) -> Optional[int]:
    """
    Extracts epoch number from model filename if it follows the pattern epoch_XXX.

    Example:
        model_epoch_011_qwk_0.8127.pth -> 11
    """

    match = re.search(r"epoch_(\d+)", path.name)
    return int(match.group(1)) if match else None


# ============================================================
# DATA
# ============================================================

def find_column(df: pd.DataFrame, options):
    """
    Finds the first matching column from a list of possible column names.
    """

    for col in options:
        if col in df.columns:
            return col

    return None


def load_test_dataframe(cfg: Config):
    """
    Loads and validates the test CSV.

    The CSV must contain:
        - an image column, for example image or filename
        - a label column, for example level or class

    Returns:
        DataFrame with standardised columns:
            image
            label
    """

    df = pd.read_csv(cfg.csv_path)

    image_col = find_column(df, cfg.image_col_options)
    label_col = find_column(df, cfg.label_col_options)

    if image_col is None or label_col is None:
        raise ValueError(
            "CSV must contain an image column such as 'image' or 'filename', "
            "and a label column such as 'level' or 'class'."
        )

    df = df[[image_col, label_col]].copy()
    df.columns = ["image", "label"]

    df["image"] = df["image"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    found_labels = set(df["label"].unique())
    valid_labels = set(range(cfg.num_classes))

    if not found_labels.issubset(valid_labels):
        raise ValueError(
            f"Found labels outside 0-{cfg.num_classes - 1}: {sorted(found_labels)}"
        )

    duplicates = df[df["image"].duplicated()]["image"].tolist()

    if duplicates:
        raise ValueError(f"Duplicate image names in CSV. Examples: {duplicates[:10]}")

    return df


def build_image_index(test_dir: Path):
    """
    Builds a mapping from image basename to full path.

    This allows the CSV to contain image names with or without file extensions.
    """

    image_index = {}
    duplicates = []

    for path in test_dir.iterdir():
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


def build_test_samples(cfg: Config):
    """
    Matches CSV rows to image files on disk.

    Returns:
        List of dictionaries containing:
            path
            image
            label
    """

    df = load_test_dataframe(cfg)
    image_index = build_image_index(cfg.test_dir)

    samples = []
    missing = []

    for _, row in df.iterrows():
        image_name = str(row["image"]).strip()
        label = int(row["label"])

        key = Path(image_name).stem.strip().lower()

        if key in image_index:
            samples.append(
                {
                    "path": image_index[key],
                    "image": image_name,
                    "label": label,
                }
            )
        else:
            missing.append(image_name)

    logging.info("\n================ TEST FILE MATCHING ================")
    logging.info(f"CSV entries        : {len(df)}")
    logging.info(f"Matched images     : {len(samples)}")
    logging.info(f"Missing images     : {len(missing)}")

    if missing:
        logging.info(f"Example missing    : {missing[:20]}")

    logging.info("====================================================\n")

    if not samples:
        raise ValueError("No test samples found. Check filename matching.")

    return samples


class TestDataset(Dataset):
    """
    Dataset for loading test images.

    Each returned item contains:
        - transformed image tensor
        - true label
        - original image name from CSV
        - full image path

    The test transform is deterministic. It resizes images to cfg.image_size,
    converts them to tensors, and applies ImageNet normalisation.
    """

    def __init__(self, samples, cfg: Config):
        self.samples = samples
        self.cfg = cfg

        self.transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        path = item["path"]
        label = item["label"]

        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {path}") from e

        image = self.transform(image)

        return image, label, item["image"], str(path)


def build_loader(cfg: Config, samples, device):
    """
    Builds the DataLoader for test inference.

    Efficiency settings:
        - pin_memory is enabled when using CUDA.
        - persistent_workers and prefetch_factor are enabled when num_workers > 0.
    """

    dataset = TestDataset(samples, cfg)

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
    }

    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **loader_kwargs)


# ============================================================
# MODEL
# ============================================================

class ParallelEfficientNetSwinCORN(nn.Module):
    """
    Dual-backbone CORN model for diabetic retinopathy grading.

    Architecture:
        - EfficientNet branch
        - Swin Transformer branch
        - Concatenation of both feature vectors
        - Fusion head
        - CORN ordinal output layer

    Important:
        This test model must match the training model exactly.
        In the training script, the final head is called fusion_head.
        Therefore, this test script also uses fusion_head.

        If this name is changed, loading a checkpoint with strict=True may fail.
    """

    def __init__(self, cfg: Config):
        super().__init__()

        self.efficientnet = timm.create_model(
            cfg.efficientnet_name,
            pretrained=False,
            num_classes=0,
        )

        self.swin = timm.create_model(
            cfg.swin_name,
            pretrained=False,
            num_classes=0,
        )

        eff_dim = getattr(self.efficientnet, "num_features", None)
        swin_dim = getattr(self.swin, "num_features", None)

        if eff_dim is None or swin_dim is None:
            raise ValueError("Could not determine feature dimensions from timm models.")

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

        Supports:
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
            # Handles both NCHW and NHWC outputs.
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


def extract_state_dict(checkpoint: Dict[str, Any]):
    """
    Extracts model weights from common PyTorch checkpoint formats.

    Supported formats:
        1. Raw state_dict:
            {"efficientnet.xxx": tensor, "swin.xxx": tensor, ...}

        2. Full checkpoint from improved training script:
            {"model_state_dict": ...}

        3. Other common formats:
            {"state_dict": ...}
            {"model": ...}
    """

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    possible_keys = ["model_state_dict", "state_dict", "model"]

    for key in possible_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    # If all values are tensors, assume the checkpoint itself is a raw state_dict.
    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise ValueError(
        "Could not find model weights in checkpoint. Expected one of: "
        "'model_state_dict', 'state_dict', 'model', or a raw state_dict."
    )


def clean_state_dict(state_dict: Dict[str, torch.Tensor]):
    """
    Cleans a state_dict before loading.

    Currently handles:
        - removing 'module.' prefix from DataParallel checkpoints
    """

    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    return cleaned


def load_model(cfg: Config, device):
    """
    Builds the model, loads checkpoint weights, moves it to device, and sets eval mode.
    """

    model = ParallelEfficientNetSwinCORN(cfg)

    checkpoint = torch.load(cfg.model_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    state_dict = clean_state_dict(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


# ============================================================
# EVALUATION
# ============================================================

def calculate_metrics(targets, preds, avg_loss, cfg: Config):
    """
    Calculates final test-set metrics.

    Metrics:
        - CORN loss
        - Accuracy
        - Quadratic Weighted Kappa
        - Micro F1
        - Weighted F1
        - Macro F1
        - Per-class F1
        - Per-class recall
    """

    qwk = cohen_kappa_score(
        targets,
        preds,
        labels=list(range(cfg.num_classes)),
        weights="quadratic",
    )

    if np.isnan(qwk):
        qwk = 0.0

    return {
        "test_corn_loss": float(avg_loss),
        "accuracy": float(accuracy_score(targets, preds)),
        "qwk": float(qwk),
        "micro_f1": float(f1_score(targets, preds, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "f1_per_class": f1_score(
            targets,
            preds,
            average=None,
            labels=list(range(cfg.num_classes)),
            zero_division=0,
        ),
        "recall_per_class": recall_score(
            targets,
            preds,
            average=None,
            labels=list(range(cfg.num_classes)),
            zero_division=0,
        ),
    }


def evaluate(model, loader, cfg: Config, device):
    """
    Runs inference on the full test set.

    Efficiency:
        - torch.inference_mode disables gradient tracking.
        - autocast is enabled on CUDA for faster mixed precision inference.

    Returns:
        metrics:
            Dictionary of test metrics.

        predictions_df:
            DataFrame containing image-level predictions.
    """

    use_amp = device.type == "cuda"

    all_preds = []
    all_targets = []
    all_images = []
    all_paths = []

    total_loss = 0.0

    with torch.inference_mode():
        for images, labels, image_names, paths in tqdm(loader, desc="Testing"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = corn_loss(outputs.float(), labels, num_classes=cfg.num_classes)

            preds = corn_label_from_logits(outputs.float())

            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_images.extend(list(image_names))
            all_paths.extend(list(paths))

    avg_loss = total_loss / max(1, len(loader))

    metrics = calculate_metrics(
        targets=all_targets,
        preds=all_preds,
        avg_loss=avg_loss,
        cfg=cfg,
    )

    predictions_df = pd.DataFrame({
        "image": all_images,
        "path": all_paths,
        "true_label": all_targets,
        "predicted_label": all_preds,
        "correct": np.array(all_targets) == np.array(all_preds),
    })

    return metrics, predictions_df


# ============================================================
# OUTPUTS
# ============================================================

def save_metrics_csv(metrics, cfg: Config):
    """
    Saves test-level metrics to CSV.
    """

    row = {
        "test_corn_loss": metrics["test_corn_loss"],
        "accuracy": metrics["accuracy"],
        "qwk": metrics["qwk"],
        "micro_f1": metrics["micro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_f1": metrics["macro_f1"],
    }

    for i, value in enumerate(metrics["f1_per_class"]):
        row[f"f1_class_{i}"] = float(value)

    for i, value in enumerate(metrics["recall_per_class"]):
        row[f"recall_class_{i}"] = float(value)

    pd.DataFrame([row]).to_csv(cfg.metrics_csv_path, index=False)


def save_predictions_csv(predictions_df, cfg: Config):
    """
    Saves image-level predictions to CSV.
    """

    predictions_df.to_csv(cfg.predictions_csv_path, index=False)


def save_confusion_matrix(targets, preds, cfg: Config):
    """
    Saves a confusion matrix image for the test predictions.
    """

    cm = confusion_matrix(
        targets,
        preds,
        labels=list(range(cfg.num_classes)),
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix\n{cfg.experiment_name}")
    plt.colorbar()

    ticks = list(range(cfg.num_classes))
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)

    for i in range(cfg.num_classes):
        for j in range(cfg.num_classes):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(cfg.confusion_matrix_path, dpi=200, bbox_inches="tight")
    plt.close()


def log_config(cfg: Config):
    """
    Logs the test configuration.
    """

    logging.info("=" * 80)
    logging.info("TEST RUN CONFIG")
    logging.info("=" * 80)

    for key, value in asdict(cfg).items():
        logging.info(f"{key}: {value}")

    logging.info("=" * 80)


def log_metrics(metrics, cfg: Config, loaded_epoch):
    """
    Logs final test metrics and output file paths.
    """

    logging.info("\n================ TEST RESULTS ================")
    logging.info(f"Experiment      : {cfg.experiment_name}")
    logging.info(f"Loaded epoch    : {loaded_epoch if loaded_epoch is not None else 'Unknown'}")
    logging.info(f"Test CORN Loss  : {metrics['test_corn_loss']:.4f}")
    logging.info(f"Accuracy        : {metrics['accuracy']:.4f}")
    logging.info(f"QWK             : {metrics['qwk']:.4f}")
    logging.info(f"Micro F1        : {metrics['micro_f1']:.4f}")
    logging.info(f"Weighted F1     : {metrics['weighted_f1']:.4f}")
    logging.info(f"Macro F1        : {metrics['macro_f1']:.4f}")

    f1_text = ", ".join(
        f"C{i}={value:.4f}" for i, value in enumerate(metrics["f1_per_class"])
    )

    recall_text = ", ".join(
        f"C{i}={value:.4f}" for i, value in enumerate(metrics["recall_per_class"])
    )

    logging.info(f"Per-class F1    : {f1_text}")
    logging.info(f"Per-class Recall: {recall_text}")
    logging.info("==============================================\n")

    logging.info(f"Metrics CSV saved to      : {cfg.metrics_csv_path}")
    logging.info(f"Predictions CSV saved to  : {cfg.predictions_csv_path}")
    logging.info(f"Confusion matrix saved to : {cfg.confusion_matrix_path}")
    logging.info(f"Test log saved to         : {cfg.log_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main entry point for the test script.
    """

    make_dirs(CFG)
    setup_logging(CFG)
    validate_config(CFG)
    check_paths(CFG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_epoch = parse_epoch_from_path(CFG.model_path)

    log_config(CFG)

    logging.info(f"Using device: {device}")
    logging.info(f"Loading model from: {CFG.model_path}")

    samples = build_test_samples(CFG)
    loader = build_loader(CFG, samples, device)

    model = load_model(CFG, device)

    metrics, predictions_df = evaluate(
        model=model,
        loader=loader,
        cfg=CFG,
        device=device,
    )

    save_metrics_csv(metrics, CFG)
    save_predictions_csv(predictions_df, CFG)

    save_confusion_matrix(
        targets=predictions_df["true_label"].tolist(),
        preds=predictions_df["predicted_label"].tolist(),
        cfg=CFG,
    )

    log_metrics(metrics, CFG, loaded_epoch)


if __name__ == "__main__":
    main()
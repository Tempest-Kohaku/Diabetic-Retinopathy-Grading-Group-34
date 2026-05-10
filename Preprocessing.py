# ======================================================================
# pip install opencv-python-headless pandas scikit-learn tqdm
# ======================================================================

"""
Image preprocessing script for diabetic retinopathy grading.

This script prepares retinal fundus images for model training and testing.

Main preprocessing steps:
    1. Read image using OpenCV.
    2. Optionally crop the visible fundus region.
    3. Pad image to a square shape.
    4. Resize image to the configured size (I have used 384x384).
    5. Apply CLAHE to improve local contrast.
    6. Save processed images.

Optional functionality:
    - Create stratified train/test folders using labels from a CSV file.
    - Save train/test label CSV files.
    - Save debug images showing each preprocessing step.
    - Process images in parallel for faster preprocessing.

Recommended for diabetic retinopathy grading:
    crop_mode = "auto" if images contain large black borders.
    crop_mode = "none" if images are already cropped and centred.

This script is intended for research and machine learning preprocessing.
"""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    """
    Central configuration for the preprocessing pipeline.

    This keeps all important settings in one place, making the script easier
    to reproduce, maintain and extend.
    """

    # Folder containing input images
    input_dir: Path = Path(r"C:\Path")

    # Output folder
    output_dir: Path = Path(r"C:\Path")

    # Optional CSV for stratified split
    # Required only if do_stratified_split=True
    csv_path: Path | None = None
    image_col: str = "image"
    label_col: str = "level"

    # If True, creates train/test folders using labels from CSV
    do_stratified_split: bool = False
    test_size: float = 0.17
    random_state: int = 3

    # Image processing
    image_size: tuple[int, int] = (384, 384)

    # crop_mode options:
    # "none" = do not crop again; only pad to square, resize, CLAHE
    # "auto" = automatically detect fundus area, crop, pad to square, resize, CLAHE
    crop_mode: str = "auto"

    # Auto-crop settings
    crop_threshold: int = 15
    crop_margin: float = 0.02

    # CLAHE settings
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)

    # Output settings
    output_ext: str = ".jpg"
    keep_original_name: bool = True
    overwrite: bool = True

    # Saves original, cropped/padded, resized and CLAHE images for checking
    save_debug_steps: bool = False

    # Parallel processing
    # 1 = single-threaded
    # More than 1 can speed up large datasets
    num_workers: int = 1

    @property
    def processing_log_path(self):
        return self.output_dir / "processing_log.csv"

    @property
    def split_manifest_path(self):
        return self.output_dir / "split_manifest.csv"

    @property
    def train_labels_path(self):
        return self.output_dir / "train_labels.csv"

    @property
    def test_labels_path(self):
        return self.output_dir / "test_labels.csv"


CFG = Config()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================
# SETUP AND VALIDATION
# ============================================================

def setup_logging():
    """
    Sets up simple terminal logging.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def validate_config(cfg: Config):
    """
    Validates preprocessing configuration before processing starts.

    This catches common mistakes early, such as invalid crop mode, invalid
    image size, invalid output extension or missing CSV path for stratified
    splitting.
    """

    valid_crop_modes = {"none", "auto"}

    if cfg.crop_mode not in valid_crop_modes:
        raise ValueError(
            f"Invalid crop_mode='{cfg.crop_mode}'. "
            f"Choose from: {sorted(valid_crop_modes)}"
        )

    if not cfg.input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {cfg.input_dir}")

    if cfg.image_size[0] <= 0 or cfg.image_size[1] <= 0:
        raise ValueError("image_size values must be positive.")

    if cfg.clahe_clip_limit <= 0:
        raise ValueError("clahe_clip_limit must be greater than 0.")

    if cfg.clahe_tile_grid_size[0] <= 0 or cfg.clahe_tile_grid_size[1] <= 0:
        raise ValueError("clahe_tile_grid_size values must be positive.")

    if not cfg.output_ext.startswith("."):
        raise ValueError("output_ext must start with '.', for example '.jpg'.")

    if cfg.num_workers < 1:
        raise ValueError("num_workers must be at least 1.")

    if cfg.do_stratified_split:
        if cfg.csv_path is None:
            raise ValueError("csv_path is required when do_stratified_split=True.")

        if not cfg.csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {cfg.csv_path}")

        if not 0 < cfg.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")


def log_config(cfg: Config):
    """
    Logs all configuration values.
    """

    logging.info("=" * 80)
    logging.info("PREPROCESSING CONFIG")
    logging.info("=" * 80)

    for key, value in asdict(cfg).items():
        logging.info(f"{key}: {value}")

    logging.info("=" * 80)


# ============================================================
# BASIC IMAGE HELPERS
# ============================================================

def read_image_bgr(path: Path):
    """
    Reads an image using OpenCV in BGR colour format.

    Raises:
        ValueError if the image cannot be read.
    """

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")

    return image


def save_image(path: Path, image_bgr):
    """
    Saves an image using OpenCV.

    Parent folders are created automatically.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image_bgr)

    if not success:
        raise ValueError(f"Could not save image: {path}")


def image_key(path_or_name):
    """
    Creates a standard matching key from an image filename or path.

    Example:
        '12345.jpg' -> '12345'
    """

    return Path(str(path_or_name)).stem.strip().lower()


# ============================================================
# PREPROCESSING FUNCTIONS
# ============================================================

def pad_to_square(image_bgr, fill_value=(0, 0, 0)):
    """
    Pads an image to a square without stretching it.

    This is important because directly resizing a rectangular fundus image
    to 384x384 can make the retina look oval.
    """

    h, w = image_bgr.shape[:2]
    square_size = max(h, w)

    pad_top = (square_size - h) // 2
    pad_bottom = square_size - h - pad_top
    pad_left = (square_size - w) // 2
    pad_right = square_size - w - pad_left

    square = cv2.copyMakeBorder(
        image_bgr,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=fill_value,
    )

    return square


def auto_crop_fundus(image_bgr, threshold=15, margin=0.02):
    """
    Automatically detects the visible fundus area and crops around it.

    This is useful for raw fundus images with large black borders.

    If no fundus-like region is detected, the original image is returned.
    """

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(gray_blur, threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return image_bgr

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    img_h, img_w = image_bgr.shape[:2]
    margin_px = int(max(w, h) * margin)

    x1 = max(0, x - margin_px)
    y1 = max(0, y - margin_px)
    x2 = min(img_w, x + w + margin_px)
    y2 = min(img_h, y + h + margin_px)

    cropped = image_bgr[y1:y2, x1:x2]

    if cropped.size == 0:
        return image_bgr

    return cropped


def resize_image(image_bgr, size=(384, 384)):
    """
    Resizes an image to the target size using area interpolation.
    """

    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)


def apply_clahe_bgr(image_bgr, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE to the L channel in LAB colour space.

    This improves local contrast while preserving colour better than applying
    CLAHE separately to RGB/BGR channels.
    """

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size,
    )

    l_clahe = clahe.apply(l_channel)

    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    output_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return output_bgr


def preprocess_image(image_path: Path, cfg: Config):
    """
    Applies the full preprocessing pipeline.

    Processing order:
        original image
            -> optional fundus crop
            -> pad to square
            -> resize to cfg.image_size
            -> CLAHE

    Padding to square before resizing prevents distorted/oval fundus images.
    """

    original = read_image_bgr(image_path)

    if cfg.crop_mode == "auto":
        cropped = auto_crop_fundus(
            original,
            threshold=cfg.crop_threshold,
            margin=cfg.crop_margin,
        )
    elif cfg.crop_mode == "none":
        cropped = original
    else:
        raise ValueError("crop_mode must be either 'none' or 'auto'.")

    square = pad_to_square(cropped)
    resized = resize_image(square, cfg.image_size)

    clahe = apply_clahe_bgr(
        resized,
        clip_limit=cfg.clahe_clip_limit,
        tile_grid_size=cfg.clahe_tile_grid_size,
    )

    return {
        "original": original,
        "cropped_or_input": cropped,
        "square": square,
        "resized": resized,
        "clahe": clahe,
    }


# ============================================================
# DATA COLLECTION
# ============================================================

def collect_images(input_dir: Path):
    """
    Recursively collects all supported image files from the input directory.
    """

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    image_paths = [
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]

    return sorted(image_paths)


def load_labels(cfg: Config):
    """
    Loads labels from a CSV file.

    Returns:
        Dictionary mapping image key to label.

    Example:
        {"12345": 2}
    """

    if cfg.csv_path is None:
        return None

    if not cfg.csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {cfg.csv_path}")

    df = pd.read_csv(cfg.csv_path)

    required_cols = {cfg.image_col, cfg.label_col}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    df = df[[cfg.image_col, cfg.label_col]].copy()
    df[cfg.image_col] = df[cfg.image_col].astype(str).str.strip()
    df[cfg.label_col] = df[cfg.label_col].astype(int)

    duplicates = df[df[cfg.image_col].duplicated()][cfg.image_col].tolist()

    if duplicates:
        raise ValueError(f"Duplicate image names found in CSV. Examples: {duplicates[:10]}")

    label_map = {
        image_key(row[cfg.image_col]): int(row[cfg.label_col])
        for _, row in df.iterrows()
    }

    return label_map


def match_images_with_labels(image_paths, label_map):
    """
    Matches collected image files with labels from the CSV.

    Returns:
        matched:
            List of dictionaries containing path, label and key.

        missing_labels:
            List of image names that were not found in the CSV.
    """

    matched = []
    missing_labels = []

    for path in image_paths:
        key = image_key(path)

        if key in label_map:
            matched.append(
                {
                    "path": path,
                    "label": label_map[key],
                    "key": key,
                }
            )
        else:
            missing_labels.append(path.name)

    return matched, missing_labels


# ============================================================
# SPLITTING
# ============================================================

def create_records(image_paths, cfg: Config):
    """
    Creates processing records.

    If do_stratified_split=False:
        all images go directly into output_dir.

    If do_stratified_split=True:
        labels are loaded from CSV and images are split into train/test folders
        using stratified splitting.
    """

    if not cfg.do_stratified_split:
        return [
            {
                "path": path,
                "split": "all",
                "label": None,
                "key": image_key(path),
            }
            for path in image_paths
        ]

    label_map = load_labels(cfg)

    if label_map is None:
        raise ValueError(
            "Stratified split requires csv_path. "
            "Either provide a CSV file or set do_stratified_split=False."
        )

    matched, missing_labels = match_images_with_labels(image_paths, label_map)

    logging.info("\n================ LABEL MATCHING ================")
    logging.info(f"Images found       : {len(image_paths)}")
    logging.info(f"Images with labels : {len(matched)}")
    logging.info(f"Missing labels     : {len(missing_labels)}")

    if missing_labels:
        logging.info(f"Example missing labels: {missing_labels[:20]}")

    logging.info("================================================\n")

    if not matched:
        raise ValueError("No images matched labels. Check CSV image names and folder filenames.")

    labels = [item["label"] for item in matched]

    train_records, test_records = train_test_split(
        matched,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=labels,
    )

    for record in train_records:
        record["split"] = "train"

    for record in test_records:
        record["split"] = "test"

    return train_records + test_records


# ============================================================
# OUTPUT PATHS
# ============================================================

def get_output_filename(input_path: Path, cfg: Config):
    """
    Builds the output filename for a processed image.
    """

    if cfg.keep_original_name:
        return input_path.stem + cfg.output_ext

    return input_path.name


def get_output_path(record, cfg: Config):
    """
    Builds the output path for a processed image.
    """

    filename = get_output_filename(record["path"], cfg)

    if cfg.do_stratified_split:
        return cfg.output_dir / record["split"] / filename

    return cfg.output_dir / filename


def get_debug_output_paths(record, cfg: Config):
    """
    Builds output paths for optional debug images.
    """

    image_stem = record["path"].stem
    split = record["split"] if cfg.do_stratified_split else "all"

    debug_dir = cfg.output_dir / "debug_steps" / split / image_stem

    return {
        "original": debug_dir / f"{image_stem}_01_original.jpg",
        "cropped_or_input": debug_dir / f"{image_stem}_02_cropped_or_input.jpg",
        "square": debug_dir / f"{image_stem}_03_square_padded.jpg",
        "resized": debug_dir / f"{image_stem}_04_resized_{cfg.image_size[0]}x{cfg.image_size[1]}.jpg",
        "clahe": debug_dir / f"{image_stem}_05_clahe.jpg",
    }


# ============================================================
# PROCESSING
# ============================================================

def process_record(record, cfg: Config):
    """
    Processes one image record.

    Returns:
        Dictionary containing processing status for the log file.
    """

    input_path = record["path"]
    output_path = get_output_path(record, cfg)

    if output_path.exists() and not cfg.overwrite:
        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "split": record["split"],
            "label": record["label"],
            "status": "skipped_exists",
            "error": "",
        }

    processed = preprocess_image(input_path, cfg)

    save_image(output_path, processed["clahe"])

    if cfg.save_debug_steps:
        debug_paths = get_debug_output_paths(record, cfg)

        for step_name, step_path in debug_paths.items():
            save_image(step_path, processed[step_name])

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "split": record["split"],
        "label": record["label"],
        "status": "processed",
        "error": "",
    }


def process_record_safe(record, cfg: Config):
    """
    Safe wrapper around process_record.

    This prevents a single failed image from stopping the full preprocessing run.
    """

    try:
        return process_record(record, cfg)

    except Exception as e:
        return {
            "input_path": str(record["path"]),
            "output_path": "",
            "split": record.get("split", ""),
            "label": record.get("label", ""),
            "status": "failed",
            "error": str(e),
        }


def process_records(records, cfg: Config):
    """
    Processes all image records.

    Uses single-threaded processing when cfg.num_workers == 1.
    Uses ThreadPoolExecutor when cfg.num_workers > 1.
    """

    if cfg.num_workers == 1:
        return [
            process_record_safe(record, cfg)
            for record in tqdm(records, desc="Preprocessing images")
        ]

    results = []

    with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
        future_to_record = {
            executor.submit(process_record_safe, record, cfg): record
            for record in records
        }

        for future in tqdm(
            as_completed(future_to_record),
            total=len(future_to_record),
            desc="Preprocessing images",
        ):
            results.append(future.result())

    return results


def save_split_csv(records, cfg: Config):
    """
    Saves train/test CSV files when stratified splitting is enabled.
    """

    if not cfg.do_stratified_split:
        return

    rows = []

    for record in records:
        output_path = get_output_path(record, cfg)

        rows.append(
            {
                "image": output_path.name,
                "path": str(output_path),
                "split": record["split"],
                "level": record["label"],
            }
        )

    split_df = pd.DataFrame(rows)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    split_df.to_csv(cfg.split_manifest_path, index=False)
    split_df[split_df["split"] == "train"].to_csv(
        cfg.train_labels_path,
        index=False,
    )
    split_df[split_df["split"] == "test"].to_csv(
        cfg.test_labels_path,
        index=False,
    )


def run_preprocessing(cfg: Config):
    """
    Runs the full preprocessing pipeline.
    """

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(cfg.input_dir)

    logging.info("\n================ INPUT IMAGES ================")
    logging.info(f"Input folder : {cfg.input_dir}")
    logging.info(f"Output folder: {cfg.output_dir}")
    logging.info(f"Images found : {len(image_paths)}")
    logging.info(f"Crop mode    : {cfg.crop_mode}")
    logging.info(f"Image size   : {cfg.image_size}")
    logging.info(f"Workers      : {cfg.num_workers}")
    logging.info("==============================================\n")

    if not image_paths:
        raise FileNotFoundError(f"No image files found in: {cfg.input_dir}")

    records = create_records(image_paths, cfg)

    results = process_records(records, cfg)

    log_df = pd.DataFrame(results)
    log_df.to_csv(cfg.processing_log_path, index=False)

    save_split_csv(records, cfg)

    processed_count = int((log_df["status"] == "processed").sum())
    skipped_count = int((log_df["status"] == "skipped_exists").sum())
    failed_count = int((log_df["status"] == "failed").sum())

    logging.info("\n================ DONE ================")
    logging.info(f"Output folder  : {cfg.output_dir}")
    logging.info(f"Processed      : {processed_count}")
    logging.info(f"Skipped        : {skipped_count}")
    logging.info(f"Failed         : {failed_count}")
    logging.info(f"Processing log : {cfg.processing_log_path}")

    if cfg.do_stratified_split:
        logging.info(f"Train folder   : {cfg.output_dir / 'train'}")
        logging.info(f"Test folder    : {cfg.output_dir / 'test'}")
        logging.info(f"Train CSV      : {cfg.train_labels_path}")
        logging.info(f"Test CSV       : {cfg.test_labels_path}")

    if cfg.save_debug_steps:
        logging.info(f"Debug folder   : {cfg.output_dir / 'debug_steps'}")

    if failed_count > 0:
        logging.warning("\nWarning: some images failed. Check processing_log.csv.")

    logging.info("======================================\n")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Main entry point for the preprocessing script.
    """

    setup_logging()
    validate_config(CFG)
    log_config(CFG)
    run_preprocessing(CFG)


if __name__ == "__main__":
    main()
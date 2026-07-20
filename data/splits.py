"""
splits.py

Scans data/raw/<class_name>/*.jpg (one folder per class), builds a
stratified train/val/test split, and writes out:

    data/splits/train.csv
    data/splits/val.csv
    data/splits/test.csv
    data/class_names.json

class_names.json stores a FIXED class_to_idx mapping, built once here
and reused everywhere else (dataset.py, evaluate.py, etc). This matters
because if train/val/test each inferred their own mapping independently,
class index 7 could mean a different animal in each split.

All paths and split ratios come from config.py -- nothing is hardcoded
here so there's a single source of truth for the whole pipeline.

Run this once before training:
    python data/splits.py
"""

import json
import random
from pathlib import Path

import pandas as pd

import config


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_image_paths(raw_dir):
    """
    Walks raw_dir/<class_name>/*.jpg and returns a list of
    (filepath, class_name) tuples.
    """
    raw_dir = Path(raw_dir)
    samples = []
    class_names = sorted([d.name for d in raw_dir.iterdir() if d.is_dir()])

    if not class_names:
        raise ValueError(
            f"No class folders found in {raw_dir}. "
            f"Expected structure: {raw_dir}/<class_name>/*.jpg"
        )

    for class_name in class_names:
        class_dir = raw_dir / class_name
        image_paths = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        ]

        if len(image_paths) == 0:
            print(f"WARNING: no images found for class '{class_name}', skipping.")
            continue

        for path in image_paths:
            samples.append((str(path), class_name))

    return samples, class_names


def stratified_split(samples, train_ratio, val_ratio, test_ratio, seed):
    """
    Splits samples into train/val/test, keeping class proportions
    consistent across all three sets (stratified by class).

    Args:
        samples: list of (filepath, class_name) tuples

    Returns:
        train_samples, val_samples, test_samples -- each a list of
        (filepath, class_name) tuples
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train/val/test ratios must sum to 1.0"

    random.seed(seed)

    # group samples by class so each class is split independently
    by_class = {}
    for filepath, class_name in samples:
        by_class.setdefault(class_name, []).append(filepath)

    train_samples, val_samples, test_samples = [], [], []

    for class_name, filepaths in by_class.items():
        filepaths = filepaths.copy()
        random.shuffle(filepaths)

        n = len(filepaths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # remainder goes to test, avoids rounding errors dropping samples

        train_files = filepaths[:n_train]
        val_files = filepaths[n_train:n_train + n_val]
        test_files = filepaths[n_train + n_val:]

        train_samples.extend((f, class_name) for f in train_files)
        val_samples.extend((f, class_name) for f in val_files)
        test_samples.extend((f, class_name) for f in test_files)

    return train_samples, val_samples, test_samples


def save_csv(samples, out_path):
    """Writes a list of (filepath, class_name) tuples to a CSV."""
    df = pd.DataFrame(samples, columns=["filepath", "label"])
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} samples to {out_path}")


def main(force=False):
    """
    Builds the train/val/test splits and class_names.json.

    Args:
        force: if False (default) and splits already exist, skips
               rebuilding them -- avoids accidentally re-shuffling
               splits (and invalidating checkpoints trained on the
               old split) on every run of main.py.
    """
    splits_dir = Path(config.SPLITS_DIR)
    train_csv = splits_dir / "train.csv"

    if train_csv.exists() and not force:
        print(f"Splits already exist at {splits_dir}, skipping. "
              f"Pass force=True to rebuild.")
        return

    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {config.RAW_DIR} for class folders...")
    samples, class_names = collect_image_paths(config.RAW_DIR)
    print(f"Found {len(class_names)} classes, {len(samples)} total images.")

    # --- build and save the FIXED class_to_idx mapping ---
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"Saved class_to_idx mapping to {config.CLASS_NAMES_PATH}")

    # --- stratified split ---
    train_samples, val_samples, test_samples = stratified_split(
        samples,
        config.TRAIN_RATIO,
        config.VAL_RATIO,
        config.TEST_RATIO,
        seed=config.SPLIT_SEED,
    )

    print(f"\nSplit sizes:")
    print(f"  train: {len(train_samples)}")
    print(f"  val:   {len(val_samples)}")
    print(f"  test:  {len(test_samples)}")

    save_csv(train_samples, splits_dir / "train.csv")
    save_csv(val_samples, splits_dir / "val.csv")
    save_csv(test_samples, splits_dir / "test.csv")

    # --- sanity check: print per-class counts for train split ---
    train_df = pd.DataFrame(train_samples, columns=["filepath", "label"])
    print("\nPer-class training sample counts:")
    print(train_df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()

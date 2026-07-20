"""
dataset.py

Two things live here:

1. ImageDataset -- reads the train/val/test CSVs produced by
   data/splits.py, loads images from disk, applies transforms, and
   returns (image, label) pairs.

2. PKSampler -- a custom batch sampler that builds each batch from
   P classes x K images per class. Triplet mining in losses.py needs
   multiple samples per class in the same batch to find meaningful
   hardest-positive / hardest-negative pairs -- a randomly shuffled
   DataLoader would rarely group same-class samples together.

Image size comes from config.IMAGE_SIZE, so switching between a
from-scratch backbone (small images, e.g. 128) and a pretrained one
(typically 224) is controlled from config.py, no code changes needed.
"""

import json
import random

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T

import config


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_train_transform(image_size):
    """
    Training from scratch (no pretrained weights) needs strong augmentation
    to make up for the small dataset size (~300 images/class). These same
    augmentations are also fine to use with pretrained backbones -- just
    applied at whatever image_size the backbone expects.
    """
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.25),
    ])


def build_eval_transform(image_size):
    """No augmentation at val/test time -- just resize + normalize."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# built once at import time, using whatever size is set in config.py
train_transform = build_train_transform(config.IMAGE_SIZE)
eval_transform = build_eval_transform(config.IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImageDataset(Dataset):
    """
    Reads a CSV with columns [filepath, label] and serves (image, label) pairs.

    Args:
        csv_path:         path to train.csv / val.csv / test.csv from data/splits.py
        class_names_path: path to the fixed class_to_idx mapping (data/class_names.json),
                           so train/val/test all agree on what each index means.
                           Defaults to config.CLASS_NAMES_PATH.
        train:            if True, applies train_transform (augmentation);
                           if False, applies eval_transform (no augmentation)
    """

    def __init__(self, csv_path, class_names_path=None, train=True):
        self.df = pd.read_csv(csv_path)
        self.transform = train_transform if train else eval_transform

        class_names_path = class_names_path or config.CLASS_NAMES_PATH
        with open(class_names_path, "r") as f:
            self.class_to_idx = json.load(f)

        self.labels = [self.class_to_idx[label] for label in self.df["label"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


# ---------------------------------------------------------------------------
# PK Sampler
# ---------------------------------------------------------------------------

class PKSampler(Sampler):
    """
    Builds batches of P classes x K images per class (batch_size = P * K).

    Needed for batch-hard triplet loss: each batch must contain multiple
    samples from the same class so the loss can find a hardest-positive
    (same class) and hardest-negative (different class) for every anchor.

    Args:
        labels: list/array of integer labels, same length and order as
                the dataset it will sample from
        p:      number of distinct classes per batch
        k:      number of images per class per batch
    """

    def __init__(self, labels, p=16, k=4):
        self.labels = labels
        self.p = p
        self.k = k

        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            self.label_to_indices.setdefault(label, []).append(idx)

        self.classes = list(self.label_to_indices.keys())

        if len(self.classes) < p:
            raise ValueError(
                f"PKSampler: requested p={p} classes per batch, "
                f"but dataset only has {len(self.classes)} classes."
            )

    def __iter__(self):
        num_batches = len(self.labels) // (self.p * self.k)
        for _ in range(num_batches):
            batch = []
            chosen_classes = random.sample(self.classes, self.p)
            for cls in chosen_classes:
                indices = self.label_to_indices[cls]
                if len(indices) >= self.k:
                    batch.extend(random.sample(indices, self.k))
                else:
                    # class has fewer than k images -- sample with replacement
                    batch.extend(random.choices(indices, k=self.k))
            yield batch

    def __len__(self):
        return len(self.labels) // (self.p * self.k)


if __name__ == "__main__":
    # quick sanity check using fake in-memory data (no real CSV needed)
    fake_labels = [i % 5 for i in range(100)]  # 5 fake classes, 20 samples each
    sampler = PKSampler(fake_labels, p=3, k=4)

    batch = next(iter(sampler))
    batch_labels = [fake_labels[i] for i in batch]
    print(f"Batch size: {len(batch)}")
    print(f"Labels in batch: {sorted(batch_labels)}")
    print(f"Number of batches per epoch: {len(sampler)}")
    print(f"Current config.IMAGE_SIZE: {config.IMAGE_SIZE}")

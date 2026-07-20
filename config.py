"""
config.py

Single source of truth for every path and hyperparameter used across
the pipeline. Nothing else in the codebase should hardcode a number or
path -- if you want to change batch size, learning rate, epochs, image
size, etc., change it here and it propagates everywhere.
"""

import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAW_DIR = "data/raw"                        # data/raw/<class_name>/*.jpg
SPLITS_DIR = "data/splits"                  # train.csv / val.csv / test.csv
CLASS_NAMES_PATH = "data/class_names.json"  # fixed class_to_idx mapping

CHECKPOINT_DIR = "checkpoints"              # encoder.pt, classifier.pt
RESULTS_DIR = "results"                     # metrics.json, plots, reports

# ---------------------------------------------------------------------------
# Device / hardware
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4          # dataloader workers; drop to 2 if CPU becomes a bottleneck
USE_AMP = True           # mixed precision -- roughly halves VRAM usage on 8GB cards

# ---------------------------------------------------------------------------
# Data splitting (used by data/splits.py)
# ---------------------------------------------------------------------------

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Backbone selection (must come before IMAGE_SIZE, which depends on it)
# ---------------------------------------------------------------------------

# One of: "tinyresnet", "mobilenetv3", "mobilevit_xs", "resnet18", "efficientnet_lite0"
# "tinyresnet" is the from-scratch backbone; the rest are ImageNet-pretrained
# via timm and are usually the better starting point given ~300 images/class.
BACKBONE_NAME = "resnet18"

PRETRAINED = True          # ignored when BACKBONE_NAME == "tinyresnet"
FREEZE_BACKBONE = False    # if True, only the projection head trains (fast, low VRAM)

# ---------------------------------------------------------------------------
# Image / augmentation
# ---------------------------------------------------------------------------

# 128 is a good fit for the from-scratch TinyResNet on 8GB VRAM.
# For pretrained backbones (mobilenetv3, mobilevit_xs, resnet18,
# efficientnet_lite0), 224 matches their ImageNet pretraining resolution
# and gives noticeably better results than 128. If you hit OOM at 224
# with a pretrained backbone, drop P_CLASSES or K_IMAGES first before
# dropping IMAGE_SIZE.
IMAGE_SIZE = 190 if BACKBONE_NAME != "tinyresnet" else 128

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

BASE_CHANNELS = 32       # TinyResNet width multiplier (~4-5M params at 32); ignored for pretrained backbones
EMBEDDING_DIM = 128      # dimensionality of the final L2-normalized embedding

CLASSIFIER_HIDDEN_DIM = 128
CLASSIFIER_DROPOUT = 0.3

# ---------------------------------------------------------------------------
# Stage 1: Encoder training (metric learning)
# ---------------------------------------------------------------------------

P_CLASSES = 16           # classes per batch (PKSampler)
K_IMAGES = 4             # images per class per batch -> batch size = P_CLASSES * K_IMAGES

ENCODER_LR = 3e-4
WEIGHT_DECAY = 1e-4
ENCODER_EPOCHS = 10
WARMUP_EPOCHS = 5
TRIPLET_MARGIN = 0.3

# ---------------------------------------------------------------------------
# Stage 2: Classifier training
# ---------------------------------------------------------------------------

EMBED_BATCH_SIZE = 64        # batch size for computing embeddings (no gradients -> can be larger)
CLASSIFIER_BATCH_SIZE = 64
CLASSIFIER_LR = 1e-3
CLASSIFIER_EPOCHS = 10

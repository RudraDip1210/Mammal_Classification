"""
train_encoder.py

Stage 1: trains the EmbeddingNet using batch-hard triplet loss so that
same-class images end up close together in embedding space and
different-class images end up far apart.

Exposes a single function, train_encoder(config), which main.py calls.
Saves the best checkpoint to config.CHECKPOINT_DIR/encoder.pt and
returns the trained model so Stage 2 can use it directly without
reloading from disk.
"""

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.embedding_net import EmbeddingNet
from losses import batch_hard_triplet_loss
from dataset import ImageDataset, PKSampler


def get_lr_lambda(warmup_epochs, total_epochs):
    """
    Returns a function used by LambdaLR: linear warmup for the first
    `warmup_epochs`, then cosine decay down to ~0 by `total_epochs`.
    From-scratch training is sensitive to LR early on, so warmup
    helps avoid the first few epochs blowing up the loss.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


def train_encoder(config):
    device = torch.device(config.DEVICE)
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train_dataset = ImageDataset(
        csv_path=f"{config.SPLITS_DIR}/train.csv",
        class_names_path=config.CLASS_NAMES_PATH,
        train=True,
    )
    val_dataset = ImageDataset(
        csv_path=f"{config.SPLITS_DIR}/val.csv",
        class_names_path=config.CLASS_NAMES_PATH,
        train=False,
    )

    train_sampler = PKSampler(train_dataset.labels, p=config.P_CLASSES, k=config.K_IMAGES)
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # for validation we don't need PK batches -- triplet loss still works
    # fine on a random batch for the purpose of tracking val loss, as long
    # as batch size is reasonably large
    val_sampler = PKSampler(val_dataset.labels, p=min(config.P_CLASSES, len(set(val_dataset.labels))), k=config.K_IMAGES)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # --- model ---
    model = EmbeddingNet(
        backbone_name=config.BACKBONE_NAME,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained=config.PRETRAINED,
        base_channels=config.BASE_CHANNELS,
        freeze_backbone=config.FREEZE_BACKBONE,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.ENCODER_LR,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=get_lr_lambda(warmup_epochs=config.WARMUP_EPOCHS, total_epochs=config.ENCODER_EPOCHS),
    )

    # mixed precision -- roughly halves VRAM usage, useful given 8GB budget
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

    best_val_loss = float("inf")
    encoder_path = checkpoint_dir / "encoder.pt"

    print(f"Training encoder on {device} | AMP: {config.USE_AMP}")
    print(f"Train batches/epoch: {len(train_loader)} | Val batches/epoch: {len(val_loader)}")

    for epoch in range(config.ENCODER_EPOCHS):
        # ---------------- train ----------------
        model.train()
        train_loss_total = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.ENCODER_EPOCHS} [train]")
        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                embeddings = model(images)
                loss = batch_hard_triplet_loss(embeddings, labels, margin=config.TRIPLET_MARGIN)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_total / len(train_loader)

        # ---------------- validate ----------------
        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                    embeddings = model(images)
                    loss = batch_hard_triplet_loss(embeddings, labels, margin=config.TRIPLET_MARGIN)

                val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_loader)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{config.ENCODER_EPOCHS} | "
            f"train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f} | lr: {current_lr:.6f}"
        )

        # save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), encoder_path)
            print(f"  -> New best val_loss, saved checkpoint to {encoder_path}")

    # load the best checkpoint before returning, in case the last epoch
    # wasn't the best one
    model.load_state_dict(torch.load(encoder_path, map_location=device))
    print(f"\nEncoder training complete. Best val_loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    # allows running this stage standalone: python train_encoder.py
    import config
    train_encoder(config)

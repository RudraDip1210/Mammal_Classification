"""
train_classifier.py

Stage 2: trains a small classifier head on top of the FROZEN embeddings
produced by the Stage 1 encoder. Since the encoder is frozen, embeddings
for the whole train/val set are computed once upfront -- training the
classifier itself is then fast (just an MLP on fixed vectors, no image
loading or backbone forward passes needed per epoch).

Exposes train_classifier(encoder, config), which main.py calls after
train_encoder(). Also exposes compute_embeddings(), which evaluate.py
reuses on the test set.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.classifier import ClassifierHead
from dataset import ImageDataset


@torch.no_grad()
def compute_embeddings(encoder, loader, device):
    """
    Runs the (frozen) encoder over every batch in `loader` and collects
    all embeddings and labels into single tensors.

    Args:
        encoder: trained EmbeddingNet, in eval mode
        loader:  a plain DataLoader yielding (image, label) batches
        device:  torch device

    Returns:
        embeddings: (N, embedding_dim) tensor, on CPU
        labels:     (N,) tensor, on CPU
    """
    encoder.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        embeddings = encoder(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


def train_classifier(encoder, config):
    device = torch.device(config.DEVICE)
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False  # freeze -- Stage 2 only trains the head

    # --- data: plain datasets, no PK sampler needed here ---
    train_dataset = ImageDataset(
        csv_path=f"{config.SPLITS_DIR}/train.csv",
        class_names_path=config.CLASS_NAMES_PATH,
        train=False,  # no augmentation when just computing embeddings for the head
    )
    val_dataset = ImageDataset(
        csv_path=f"{config.SPLITS_DIR}/val.csv",
        class_names_path=config.CLASS_NAMES_PATH,
        train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.EMBED_BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.EMBED_BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    print("Computing embeddings for train/val sets (frozen encoder)...")
    train_embeddings, train_labels = compute_embeddings(encoder, train_loader, device)
    val_embeddings, val_labels = compute_embeddings(encoder, val_loader, device)
    print(f"Train embeddings: {train_embeddings.shape} | Val embeddings: {val_embeddings.shape}")

    # wrap precomputed embeddings in a TensorDataset for fast training
    train_tensor_ds = TensorDataset(train_embeddings, train_labels)
    train_tensor_loader = DataLoader(
        train_tensor_ds, batch_size=config.CLASSIFIER_BATCH_SIZE, shuffle=True,
    )

    num_classes = len(train_dataset.class_to_idx)

    classifier = ClassifierHead(
        embedding_dim=config.EMBEDDING_DIM,
        num_classes=num_classes,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM,
        dropout=config.CLASSIFIER_DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.CLASSIFIER_LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    classifier_path = checkpoint_dir / "classifier.pt"

    print(f"\nTraining classifier head on {device}...")
    for epoch in range(config.CLASSIFIER_EPOCHS):
        # ---------------- train ----------------
        classifier.train()
        train_loss_total = 0.0

        for embeddings, labels in train_tensor_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()

        avg_train_loss = train_loss_total / len(train_tensor_loader)

        # ---------------- validate ----------------
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_embeddings.to(device))
            val_preds = val_logits.argmax(dim=1).cpu()
            val_acc = (val_preds == val_labels).float().mean().item()

        print(
            f"Epoch {epoch+1}/{config.CLASSIFIER_EPOCHS} | "
            f"train_loss: {avg_train_loss:.4f} | val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), classifier_path)
            print(f"  -> New best val_acc, saved checkpoint to {classifier_path}")

    classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    print(f"\nClassifier training complete. Best val_acc: {best_val_acc:.4f}")
    return classifier


if __name__ == "__main__":
    # allows running this stage standalone, loading a previously saved encoder
    import config
    from models.embedding_net import EmbeddingNet

    device = torch.device(config.DEVICE)
    encoder = EmbeddingNet(
        backbone_name=config.BACKBONE_NAME,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained=config.PRETRAINED,
        base_channels=config.BASE_CHANNELS,
        freeze_backbone=config.FREEZE_BACKBONE,
    ).to(device)
    encoder.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/encoder.pt", map_location=device))

    train_classifier(encoder, config)

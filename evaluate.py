"""
evaluate.py

Final stage: runs the trained encoder + classifier on the held-out
test set and saves everything needed to judge how well the pipeline
worked:

    results/metrics.json           -- test accuracy, per-class accuracy
    results/confusion_matrix.png   -- where the classifier gets confused
    results/tsne_embeddings.png    -- visual check that the embedding
                                       space actually separates classes
    results/classification_report.txt

Exposes evaluate_model(encoder, classifier, config), called by main.py
after both training stages finish.
"""

import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader

from dataset import ImageDataset
from train_classifier import compute_embeddings


def evaluate_model(encoder, classifier, config):
    device = torch.device(config.DEVICE)
    results_dir = Path(config.RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    encoder = encoder.to(device).eval()
    classifier = classifier.to(device).eval()

    # --- load test set ---
    test_dataset = ImageDataset(
        csv_path=f"{config.SPLITS_DIR}/test.csv",
        class_names_path=config.CLASS_NAMES_PATH,
        train=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.EMBED_BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # idx_to_class for readable labels in plots/reports
    idx_to_class = {idx: name for name, idx in test_dataset.class_to_idx.items()}
    class_names_ordered = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("Computing test set embeddings...")
    test_embeddings, test_labels = compute_embeddings(encoder, test_loader, device)

    print("Running classifier on test embeddings...")
    with torch.no_grad():
        logits = classifier(test_embeddings.to(device))
        preds = logits.argmax(dim=1).cpu()

    # ---------------- accuracy ----------------
    test_acc = accuracy_score(test_labels, preds)
    print(f"\nTest accuracy: {test_acc:.4f}")

    report_dict = classification_report(
        test_labels, preds, target_names=class_names_ordered,
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        test_labels, preds, target_names=class_names_ordered, zero_division=0,
    )

    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report_text)
    print(f"Saved classification report to {results_dir / 'classification_report.txt'}")

    # ---------------- confusion matrix ----------------
    cm = confusion_matrix(test_labels, preds)
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm, cmap="Blues", xticklabels=class_names_ordered, yticklabels=class_names_ordered,
        square=True, cbar=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (test accuracy: {test_acc:.4f})")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {results_dir / 'confusion_matrix.png'}")

    # ---------------- t-SNE of embedding space ----------------
    print("Running t-SNE on test embeddings (this can take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, init="pca")
    embeddings_2d = tsne.fit_transform(test_embeddings.numpy())

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=test_labels.numpy(), cmap="tab20", s=10, alpha=0.8,
    )
    plt.title("t-SNE of Test Set Embeddings")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")

    # build a legend mapping colors to class names (tab20 has 20 distinct
    # colors, so with 45 classes some colors repeat -- still useful visually)
    handles, _ = scatter.legend_elements(num=len(class_names_ordered))
    plt.legend(
        handles, class_names_ordered, bbox_to_anchor=(1.05, 1),
        loc="upper left", fontsize=7, ncol=2,
    )
    plt.tight_layout()
    plt.savefig(results_dir / "tsne_embeddings.png", dpi=150)
    plt.close()
    print(f"Saved t-SNE plot to {results_dir / 'tsne_embeddings.png'}")

    # ---------------- save metrics summary ----------------
    metrics = {
        "test_accuracy": test_acc,
        "macro_avg_f1": report_dict["macro avg"]["f1-score"],
        "weighted_avg_f1": report_dict["weighted avg"]["f1-score"],
        "per_class_accuracy": {
            class_names_ordered[i]: report_dict[class_names_ordered[i]]["recall"]
            for i in range(len(class_names_ordered))
        },
    }

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics summary to {results_dir / 'metrics.json'}")

    return metrics


if __name__ == "__main__":
    # allows running this stage standalone, loading previously saved
    # encoder + classifier checkpoints
    import config
    from models.embedding_net import EmbeddingNet
    from models.classifier import ClassifierHead

    device = torch.device(config.DEVICE)

    encoder = EmbeddingNet(
        backbone_name=config.BACKBONE_NAME,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained=config.PRETRAINED,
        base_channels=config.BASE_CHANNELS,
        freeze_backbone=config.FREEZE_BACKBONE,
    ).to(device)
    encoder.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/encoder.pt", map_location=device))

    # num_classes inferred the same way train_classifier.py does
    with open(config.CLASS_NAMES_PATH) as f:
        num_classes = len(json.load(f))

    classifier = ClassifierHead(
        embedding_dim=config.EMBEDDING_DIM, num_classes=num_classes,
        hidden_dim=config.CLASSIFIER_HIDDEN_DIM, dropout=config.CLASSIFIER_DROPOUT,
    ).to(device)
    classifier.load_state_dict(torch.load(f"{config.CHECKPOINT_DIR}/classifier.pt", map_location=device))

    evaluate_model(encoder, classifier, config)

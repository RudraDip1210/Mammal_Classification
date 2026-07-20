"""
losses.py

Loss functions for Stage 1 (metric learning).

Uses batch-hard triplet loss with online mining: for each anchor in
the batch, the hardest positive (farthest same-class sample) and the
hardest negative (closest different-class sample) are selected
automatically from within the batch. This is why the PKSampler in
dataset.py matters -- it guarantees each batch actually contains
multiple samples per class for mining to work.
"""

import torch
import torch.nn.functional as F


def pairwise_distances(embeddings):
    """
    Computes the pairwise squared Euclidean distance matrix for a batch
    of embeddings.

    Args:
        embeddings: (batch_size, embedding_dim), assumed L2-normalized

    Returns:
        dist: (batch_size, batch_size) matrix where dist[i, j] is the
              squared distance between embedding i and embedding j
    """
    dot_product = torch.matmul(embeddings, embeddings.t())
    squared_norm = torch.diag(dot_product)

    # ||a - b||^2 = ||a||^2 - 2*a.b + ||b||^2
    dist = squared_norm.unsqueeze(0) - 2.0 * dot_product + squared_norm.unsqueeze(1)

    # numerical safety: distances can go slightly negative due to float error
    dist = torch.clamp(dist, min=0.0)
    return dist


def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """
    Batch-hard triplet loss (Hermans et al., "In Defense of the Triplet Loss").

    For every anchor sample in the batch:
        - hardest positive = same-class sample that is FARTHEST away
        - hardest negative = different-class sample that is CLOSEST

    loss = relu(hardest_positive_dist - hardest_negative_dist + margin)

    This pushes same-class embeddings together and different-class
    embeddings apart, focusing training on the hardest examples rather
    than easy ones that contribute little signal.

    Args:
        embeddings: (batch_size, embedding_dim), L2-normalized
        labels:     (batch_size,) integer class labels
        margin:     minimum desired gap between positive and negative distances

    Returns:
        scalar loss (mean over the batch)
    """
    device = embeddings.device
    dist = pairwise_distances(embeddings)

    labels = labels.unsqueeze(1).to(device)          # (batch, 1)
    same_class_mask = labels == labels.t()             # (batch, batch), True where same class
    diff_class_mask = ~same_class_mask

    # --- hardest positive: max distance among same-class pairs ---
    # zero-out (mask) all non-same-class distances before taking max
    dist_for_positives = dist.clone()
    dist_for_positives[diff_class_mask] = 0.0
    hardest_positive_dist = dist_for_positives.max(dim=1)[0]

    # --- hardest negative: min distance among different-class pairs ---
    # set same-class distances (including self, distance=0) to +inf so they're never picked as "closest"
    dist_for_negatives = dist.clone()
    dist_for_negatives[same_class_mask] = float("inf")
    hardest_negative_dist = dist_for_negatives.min(dim=1)[0]

    triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + margin)
    return triplet_loss.mean()


if __name__ == "__main__":
    # quick sanity check with random embeddings and fake labels
    torch.manual_seed(0)
    embeddings = F.normalize(torch.randn(16, 128), p=2, dim=1)
    labels = torch.randint(0, 4, (16,))  # 4 fake classes

    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.3)
    print(f"Loss on random embeddings (expect it to be > 0): {loss.item():.4f}")

    # sanity check 2: if embeddings ARE the class one-hot vectors (perfect separation),
    # loss should be much lower
    perfect_embeddings = F.one_hot(labels, num_classes=4).float()
    perfect_embeddings = F.normalize(perfect_embeddings, p=2, dim=1)
    perfect_loss = batch_hard_triplet_loss(perfect_embeddings, labels, margin=0.3)
    print(f"Loss on perfectly separated embeddings (expect ~0): {perfect_loss.item():.4f}")

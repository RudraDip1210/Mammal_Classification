"""
classifier.py

A small classification head trained in Stage 2, on top of the frozen
embeddings produced by EmbeddingNet. Takes an embedding vector (not
an image) and outputs class logits over all animal classes.

Deliberately small -- the heavy lifting of learning good features
already happened in Stage 1.
"""

import torch.nn as nn


class ClassifierHead(nn.Module):
    """
    embedding -> small MLP -> class logits

    Args:
        embedding_dim: must match the EmbeddingNet's output dimension
        num_classes:   number of animal classes (45 in this project)
        hidden_dim:    size of the hidden layer
        dropout:       dropout rate, helps given the small dataset size
    """

    def __init__(self, embedding_dim=128, num_classes=45, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)  # raw logits -- use CrossEntropyLoss during training


if __name__ == "__main__":
    import torch

    model = ClassifierHead(embedding_dim=128, num_classes=45)
    dummy_embeddings = torch.randn(8, 128)
    logits = model(dummy_embeddings)

    print(f"Logits shape: {logits.shape}")  # expect (8, 45)
    preds = logits.argmax(dim=1)
    print(f"Predicted classes: {preds}")

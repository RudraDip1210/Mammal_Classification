"""
embedding_net.py

Wraps a backbone (from backbone.py) with a projection head to produce
L2-normalized embeddings for metric learning (Stage 1 of the pipeline).

The backbone is selected by name via get_backbone() -- config.BACKBONE_NAME
controls which one gets used (tinyresnet, mobilenetv3, mobilevit_xs,
resnet18, efficientnet_lite0). No other code needs to change when you
switch backbones.
"""

import torch.nn as nn
import torch.nn.functional as F

from models.backbone import get_backbone


class EmbeddingNet(nn.Module):
    """
    backbone -> projection head -> L2-normalized embedding

    Args:
        backbone_name: one of "tinyresnet", "mobilenetv3", "mobilevit_xs",
                       "resnet18", "efficientnet_lite0"
        embedding_dim: dimensionality of the final embedding vector
        pretrained:    whether to load ImageNet weights (ignored for tinyresnet)
        base_channels: only used when backbone_name == "tinyresnet"
        freeze_backbone: if True, backbone weights are frozen and only the
                         projection head is trained. Useful for a fast,
                         low-VRAM first pass with a pretrained backbone.
    """

    def __init__(
        self,
        backbone_name="tinyresnet",
        embedding_dim=128,
        pretrained=True,
        base_channels=32,
        freeze_backbone=False,
    ):
        super().__init__()

        self.backbone = get_backbone(
            backbone_name, pretrained=pretrained, base_channels=base_channels
        )
        feat_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.projection(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


if __name__ == "__main__":
    import torch

    # quick check across all backbones
    for backbone_name in ["tinyresnet", "mobilenetv3", "mobilevit_xs", "resnet18", "efficientnet_lite0"]:
        print(f"\n--- EmbeddingNet with {backbone_name} ---")
        model = EmbeddingNet(backbone_name=backbone_name, embedding_dim=128)
        dummy_input = torch.randn(8, 3, 128, 128)
        embeddings = model(dummy_input)
        print(f"Embedding shape: {embeddings.shape}")
        norms = embeddings.norm(p=2, dim=1)
        print(f"Sample norms (should be ~1.0): {norms[:3]}")

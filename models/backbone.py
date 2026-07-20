"""
backbone.py

Two families of backbones live here:

1. TinyResNet -- the from-scratch CNN built earlier (no pretrained weights).
2. PretrainedBackbone -- a thin wrapper around timm models, giving access
   to MobileNetV3, MobileViT-XS, ResNet18, and EfficientNet-Lite0, all
   pretrained on ImageNet.

get_backbone(name, pretrained) is the single factory function everything
else (embedding_net.py) calls. Switching backbones is just a matter of
changing config.BACKBONE_NAME -- no other file needs to change.
"""

import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------------------
# From-scratch backbone (unchanged from before)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU. Used as the initial stem layer."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Standard residual block: two 3x3 convs with a skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return self.relu(out)


class TinyResNet(nn.Module):
    """
    A compact ResNet-style backbone trained from scratch.

    Input:  (batch, 3, 128, 128)
    Output: (batch, num_features)
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()

        self.stem = ConvBlock(in_channels, base_channels, stride=2)

        self.stage1 = nn.Sequential(
            ResidualBlock(base_channels, base_channels * 2, stride=2),
            ResidualBlock(base_channels * 2, base_channels * 2),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualBlock(base_channels * 4, base_channels * 4),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 8, stride=2),
            ResidualBlock(base_channels * 8, base_channels * 8),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(base_channels * 8, base_channels * 16, stride=2),
            ResidualBlock(base_channels * 16, base_channels * 16),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_features = base_channels * 16

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return x.flatten(1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Pretrained backbones (via timm)
# ---------------------------------------------------------------------------

# maps our own short names -> actual timm model names
TIMM_MODEL_NAMES = {
    "mobilenetv3": "mobilenetv3_small_100",
    "mobilevit_xs": "mobilevit_xs",
    "resnet18": "resnet18",
    "efficientnet_lite0": "tf_efficientnet_lite0",
}


class PretrainedBackbone(nn.Module):
    """
    Thin wrapper around a timm model, used as a pure feature extractor.

    num_classes=0 strips the classification head off the timm model,
    so forward() returns pooled features directly -- same contract as
    TinyResNet (image in, flat feature vector out).
    """

    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # remove classifier head
            global_pool="avg",   # ensure output is already pooled to (batch, features)
        )
        self.num_features = self.model.num_features

    def forward(self, x):
        return self.model(x)  # (batch, num_features)


# ---------------------------------------------------------------------------
# Factory function -- the single entry point used by embedding_net.py
# ---------------------------------------------------------------------------

def get_backbone(name, pretrained=True, base_channels=32):
    """
    Builds and returns a backbone by name.

    Args:
        name: one of "tinyresnet", "mobilenetv3", "mobilevit_xs",
              "resnet18", "efficientnet_lite0"
        pretrained: only used for the timm-based backbones; ignored
                    for "tinyresnet" (which is always trained from scratch)
        base_channels: only used for "tinyresnet"

    Returns:
        an nn.Module with a `.num_features` attribute and a forward()
        that maps (batch, 3, H, W) -> (batch, num_features)
    """
    name = name.lower()

    if name == "tinyresnet":
        return TinyResNet(base_channels=base_channels)

    if name in TIMM_MODEL_NAMES:
        timm_name = TIMM_MODEL_NAMES[name]
        return PretrainedBackbone(timm_name, pretrained=pretrained)

    raise ValueError(
        f"Unknown backbone name '{name}'. "
        f"Choose from: tinyresnet, {', '.join(TIMM_MODEL_NAMES.keys())}"
    )


if __name__ == "__main__":
    # sanity check every backbone: run a dummy batch through, print shape + param count
    dummy_input = torch.randn(4, 3, 128, 128)

    for backbone_name in ["tinyresnet"] + list(TIMM_MODEL_NAMES.keys()):
        print(f"\n--- {backbone_name} ---")
        try:
            model = get_backbone(backbone_name, pretrained=True)
            output = model(dummy_input)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"Output shape: {output.shape}")
            print(f"num_features: {model.num_features}")
            print(f"Total parameters: {num_params:,}")
        except Exception as e:
            print(f"FAILED: {e}")

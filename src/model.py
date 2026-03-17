"""
model.py — EfficientNetV2-S with a custom classifier head.

Architecture:
  EfficientNetV2-S backbone (ImageNet pretrained)
    → GlobalAvgPool (built into timm)
    → Dropout(0.3)
    → Linear(num_features → num_classes)
"""

import timm
import torch
import torch.nn as nn


class EfficientNetV2Classifier(nn.Module):
    """EfficientNetV2-S with a custom dropout + linear head."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.3):
        super().__init__()
        # Load backbone; set num_classes=0 to get raw feature embeddings
        self.backbone = timm.create_model(
            "efficientnetv2_s",
            pretrained=True,
            num_classes=0,   # remove default head
            global_pool="avg",
        )
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (B, num_features)
        return self.head(features)    # (B, num_classes)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (phase-1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (phase-2 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_model(num_classes: int = 6, freeze_backbone: bool = False) -> EfficientNetV2Classifier:
    """
    Build and return the classifier.

    Args:
        num_classes:      Number of output classes.
        freeze_backbone:  If True, backbone weights are frozen on creation.

    Returns:
        EfficientNetV2Classifier instance.
    """
    model = EfficientNetV2Classifier(num_classes=num_classes)
    if freeze_backbone:
        model.freeze_backbone()
    return model

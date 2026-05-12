"""
core/model.py
─────────────
ResNet-18 backbone fine-tuned for the FairFace benchmark.
Supports feature extraction + classification in one forward pass.
"""

import torch
import torch.nn as nn
from torchvision import models


class BiasAwareCNN(nn.Module):
    """
    ResNet-18 backbone with a dual-output head:
      • logits  → for CrossEntropy classification loss
      • embeddings → 128-d feature vector for bias analysis / t-SNE

    Args:
        num_classes (int): number of target classes (e.g. 2 for gender, 7 for race)
        pretrained  (bool): use ImageNet weights
        freeze_backbone (bool): freeze all layers except the final FC
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_features = backbone.fc.in_features          # 512

        # Remove the original FC so we can swap it
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Embedding head (128-d) ────────────────────────────
        self.embed_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

        # ── Classification head ───────────────────────────────
        self.classifier = nn.Linear(128, num_classes)

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)          # (B, 512, 1, 1)
        feats = feats.flatten(1)          # (B, 512)
        embeddings = self.embed_head(feats)    # (B, 128)
        logits = self.classifier(embeddings)   # (B, num_classes)
        return logits, embeddings


def build_model(num_classes: int, pretrained: bool = True,
                freeze_backbone: bool = False, device: str = "cpu") -> BiasAwareCNN:
    """Convenience factory — builds, moves to device, returns model."""
    model = BiasAwareCNN(num_classes=num_classes,
                         pretrained=pretrained,
                         freeze_backbone=freeze_backbone)
    model = model.to(device)
    return model

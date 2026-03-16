import torch.nn as nn
import torchvision.models as models


class MLPHead(nn.Module):
    """Two-layer MLP classification head with dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout_p: float = 0.7):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, in_dim * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class Classifier(nn.Module):
    """ResNet-50 backbone with MLP classification head.

    Uses ImageNet-pretrained weights (IMAGENET1K_V2) and replaces the
    final fully-connected layer with an :class:`MLPHead`.
    """

    def __init__(self, num_classes: int, dropout_p: float = 0.7):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone_feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.mlp = MLPHead(
            in_dim=backbone_feature_dim,
            out_dim=num_classes,
            dropout_p=dropout_p,
        )
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.mlp(self.backbone(x))

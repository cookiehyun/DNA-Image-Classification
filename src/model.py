import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def extract_features(self, x):
        """Extract backbone features (2048-dim) without the MLP head."""
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Cosine-similarity OOD utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_prototypes(model, dataloader, num_classes, device):
    """Compute L2-normalised class-mean prototypes from a labelled dataset.

    Also returns the per-sample max cosine similarities to prototypes,
    which can be used to derive percentile-based adaptive thresholds.

    Args:
        model: A :class:`Classifier` instance (in eval mode).
        dataloader: Yields ``(images, labels)`` batches.
        num_classes: Number of classes.
        device: Torch device.

    Returns:
        prototypes: Tensor of shape ``(num_classes, feat_dim)`` with
            unit-norm rows.
        train_sim: 1-D tensor of max cosine similarities for each
            training sample (used for percentile-based thresholding).
    """
    model.eval()
    all_feats, all_labels = [], []
    for imgs, labels in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        feats = model.extract_features(imgs)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
    all_feats = torch.cat(all_feats)
    all_labels = torch.cat(all_labels)

    all_feats = F.normalize(all_feats, dim=1)
    prototypes = torch.zeros(num_classes, all_feats.shape[1])
    for c in range(num_classes):
        mask = all_labels == c
        if mask.any():
            prototypes[c] = all_feats[mask].mean(dim=0)
    prototypes = F.normalize(prototypes, dim=1)

    # Compute training-set cosine similarities for percentile threshold.
    train_sim = (all_feats @ prototypes.T).max(dim=1).values
    return prototypes, train_sim


def percentile_threshold(train_sim, p):
    """Compute the p-th percentile of training cosine similarities.

    At deployment, a test sample with cosine similarity below this
    value is flagged as OOD.

    Args:
        train_sim: 1-D tensor of training-set max cosine similarities
            (returned by :func:`compute_prototypes`).
        p: Percentile (0--100).  Lower *p* is more conservative
            (fewer false rejections); higher *p* rejects more aggressively.

    Returns:
        Scalar threshold value.
    """
    import numpy as np
    return float(np.percentile(train_sim.numpy(), p))


@torch.no_grad()
def cosine_ood_scores(model, imgs, prototypes, device):
    """Compute cosine-similarity OOD scores for a batch of images.

    Args:
        model: A :class:`Classifier` instance (in eval mode).
        imgs: Batch tensor on *device*.
        prototypes: ``(num_classes, feat_dim)`` tensor (CPU or GPU).
        device: Torch device.

    Returns:
        max_sim: Per-sample maximum cosine similarity to any prototype.
        preds:   Per-sample predicted class (argmax of similarity).
    """
    feats = F.normalize(model.extract_features(imgs), dim=1)
    protos = prototypes.to(feats.device)
    sim = feats @ protos.T          # (B, C)
    max_sim, preds = sim.max(dim=1)
    return max_sim, preds

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import functional as F


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class ResizeWithPad:
    """Resize an image to ``target_size`` while preserving aspect ratio.

    The shorter side is padded with the ImageNet mean colour so that the
    output is always ``(target_size, target_size)``.
    """

    IMAGENET_MEAN_RGB = tuple(int(x * 255) for x in [0.485, 0.456, 0.406])

    def __init__(
        self,
        target_size: int = 224,
        fill: Optional[Tuple[int, int, int]] = None,
        padding_mode: str = "constant",
    ):
        self.target_size = target_size
        self.fill = fill or self.IMAGENET_MEAN_RGB
        self.padding_mode = padding_mode

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return img

        scale = self.target_size / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img = F.resize(img, (new_h, new_w))

        pad_w = self.target_size - new_w
        pad_h = self.target_size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        return F.pad(
            img,
            [left, top, right, bottom],
            fill=self.fill,
            padding_mode=self.padding_mode,
        )


def get_train_transform(target_size: int = 224) -> transforms.Compose:
    """Return the training augmentation pipeline."""
    return transforms.Compose([
        ResizeWithPad(target_size),
        transforms.RandomRotation(360),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(
                brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5,
            )],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform(target_size: int = 224) -> transforms.Compose:
    """Return the evaluation (no augmentation) pipeline."""
    return transforms.Compose([
        ResizeWithPad(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class UnlabeledDataset(Dataset):
    """Dataset that loads images from a flat list of file paths."""

    def __init__(self, file_paths: List[str], transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.file_paths[idx]


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class BalancedClassSampler(Sampler):
    """Sampler that yields a balanced number of samples per class each epoch.

    Oversamples minority classes and undersamples majority classes so that
    each class contributes ``samples_per_class`` examples per epoch.
    """

    def __init__(self, dataset, samples_per_class: int = 3000, num_classes: int = None):
        self.dataset = dataset
        self.samples_per_class = samples_per_class
        if num_classes is None:
            num_classes = len(set(dataset.targets))
        self.num_classes = num_classes
        self.num_samples_per_epoch = self.samples_per_class * self.num_classes

        self.class_indices = defaultdict(list)
        for idx, target in enumerate(self.dataset.targets):
            self.class_indices[target].append(idx)

    def __iter__(self):
        all_indices = []
        for c in self.class_indices:
            sampled = np.random.choice(
                self.class_indices[c],
                self.samples_per_class,
                replace=True,
            )
            all_indices.extend(sampled)
        np.random.shuffle(all_indices)
        return iter(all_indices)

    def __len__(self) -> int:
        return self.num_samples_per_epoch

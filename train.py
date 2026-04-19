"""Train a DNA image classifier.

Example usage::

    # Train on a labelled dataset (ImageFolder format)
    python train.py --data_root ./data/train --num_classes 4 --epochs 13

    # Specify GPU
    python train.py --data_root ./data/train --num_classes 4 --gpu 0

    # Use balanced sampling with 3000 samples per class
    python train.py --data_root ./data/train --num_classes 4 --samples_per_class 3000
"""

import argparse
import os
from datetime import datetime
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.model import Classifier, compute_prototypes
from src.data import get_train_transform, get_eval_transform, BalancedClassSampler


# Default layer-wise learning rates (from hyperparameter search).
DEFAULT_LR = {
    "mlp": 2.5e-05,
    "backbone_late": 5.1e-06,
    "backbone_middle": 4.1e-06,
    "backbone_early": 7.3e-07,
}


def build_optimizer(model: Classifier, lr: dict = None) -> optim.Optimizer:
    """Build AdamW optimizer with layer-wise learning rates."""
    lr = lr or DEFAULT_LR
    param_groups = [
        {
            "params": chain(
                model.backbone.conv1.parameters(),
                model.backbone.bn1.parameters(),
                model.backbone.layer1.parameters(),
            ),
            "lr": lr["backbone_early"],
        },
        {
            "params": model.backbone.layer2.parameters(),
            "lr": lr["backbone_middle"],
        },
        {
            "params": chain(
                model.backbone.layer3.parameters(),
                model.backbone.layer4.parameters(),
            ),
            "lr": lr["backbone_late"],
        },
        {
            "params": model.mlp.parameters(),
            "lr": lr["mlp"],
        },
    ]
    return optim.AdamW(param_groups)


def train(args):
    # Device setup.
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Data.
    transform = get_train_transform(target_size=args.image_size)
    dataset = ImageFolder(root=args.data_root, transform=transform)
    class_names = dataset.classes
    print(f"Classes ({len(class_names)}): {class_names}")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": min(8, os.cpu_count() or 1),
        "pin_memory": True,
    }
    if len(class_names) > 1 and args.samples_per_class > 0:
        loader_kwargs["sampler"] = BalancedClassSampler(
            dataset,
            samples_per_class=args.samples_per_class,
            num_classes=len(class_names),
        )
    else:
        loader_kwargs["shuffle"] = True

    dataloader = DataLoader(dataset, **loader_kwargs)

    # Model.
    model = Classifier(num_classes=len(class_names)).to(device)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    # Training loop.
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for imgs, lbls in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(dim=1) == lbls).sum().item()
            total += imgs.size(0)

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total
        print(f"  Loss: {avg_loss:.4f}  Accuracy: {accuracy:.2f}%")

    # Save.
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.output_dir, f"model_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save class names alongside the model for inference.
    meta_path = os.path.join(args.output_dir, f"classes_{timestamp}.txt")
    with open(meta_path, "w") as f:
        f.write("\n".join(class_names))
    print(f"Class names saved to {meta_path}")

    # Compute and save class prototypes for cosine-similarity OOD.
    model.eval()
    eval_ds = ImageFolder(root=args.data_root, transform=get_eval_transform(args.image_size))
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=min(8, os.cpu_count() or 1))
    prototypes, train_sim = compute_prototypes(model, eval_loader, len(class_names), device)
    proto_path = os.path.join(args.output_dir, f"prototypes_{timestamp}.pt")
    torch.save({"prototypes": prototypes, "train_sim": train_sim}, proto_path)
    print(f"Prototypes saved to {proto_path}")

    return model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train DNA image classifier")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to training data (ImageFolder format)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Number of classes (auto-detected if omitted)")
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--samples_per_class", type=int, default=3000,
                        help="Balanced sampling count per class (0 to disable)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU index to use (default: auto)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

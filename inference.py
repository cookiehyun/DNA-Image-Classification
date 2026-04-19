"""Run inference with a trained DNA image classifier.

Example usage::

    # Classify images and save results (softmax OOD, default)
    python inference.py \\
        --model_path ./checkpoints/model_20250101_120000.pth \\
        --classes_path ./checkpoints/classes_20250101_120000.txt \\
        --data_root ./data/test \\
        --output_dir ./results

    # With cosine-similarity OOD filtering
    python inference.py \\
        --model_path ./checkpoints/model.pth \\
        --classes_path ./checkpoints/classes.txt \\
        --prototypes_path ./checkpoints/prototypes.pt \\
        --data_root ./data/test \\
        --ood_method cosine \\
        --confidence_threshold 75
"""

import argparse
import glob
import json
import os
import shutil

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import Classifier, cosine_ood_scores, percentile_threshold
from src.data import UnlabeledDataset, get_eval_transform


SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def run_inference(args):
    # Device.
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load class names.
    with open(args.classes_path) as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"Classes ({len(class_names)}): {class_names}")

    # Load model.
    model = Classifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    # Load prototypes (required for cosine OOD).
    prototypes = None
    train_sim = None
    if args.ood_method == "cosine":
        if not args.prototypes_path or not os.path.exists(args.prototypes_path):
            raise FileNotFoundError(
                "Cosine OOD requires --prototypes_path. "
                "Run train.py first to generate prototypes."
            )
        data = torch.load(args.prototypes_path, map_location="cpu")
        if isinstance(data, dict):
            prototypes = data["prototypes"]
            train_sim = data.get("train_sim")
        else:
            prototypes = data  # legacy: bare tensor

    # Collect image paths.
    paths = sorted([
        p for p in glob.glob(os.path.join(args.data_root, "**", "*.*"), recursive=True)
        if p.lower().endswith(SUPPORTED_EXTENSIONS)
    ])
    if not paths:
        print(f"No images found in {args.data_root}")
        return

    print(f"Found {len(paths)} images")

    # Inference.
    transform = get_eval_transform(target_size=args.image_size)
    dataset = UnlabeledDataset(paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
    )

    # Determine threshold.
    if args.ood_method == "cosine":
        p = args.confidence_threshold
        if train_sim is not None:
            threshold = percentile_threshold(train_sim, p)
            print(f"Cosine OOD: percentile {p:.0f} → threshold {threshold:.4f}")
        else:
            threshold = p / 100.0
            print(f"Cosine OOD: raw threshold {threshold:.4f} (no train_sim available)")
    else:
        threshold = args.confidence_threshold / 100.0

    results = {name: [] for name in class_names}
    results["OOD"] = []

    with torch.no_grad():
        for imgs, batch_paths in tqdm(loader, desc="Inference"):
            imgs = imgs.to(device)

            if args.ood_method == "cosine":
                confs, preds = cosine_ood_scores(model, imgs, prototypes, device)
            else:
                probs = torch.softmax(model(imgs), dim=1)
                confs, preds = torch.max(probs, dim=1)

            for i in range(len(batch_paths)):
                conf = confs[i].item()
                pred_name = class_names[preds[i].item()]
                path = batch_paths[i]

                if conf < threshold:
                    results["OOD"].append({
                        "path": path,
                        "predicted": pred_name,
                        "confidence": round(conf * 100, 2),
                    })
                else:
                    results[pred_name].append(path)

    method_label = args.ood_method.capitalize()
    # Print summary.
    print(f"\n--- Classification Results ({method_label} OOD) ---")
    for cls in class_names:
        print(f"  {cls}: {len(results[cls])} images")
    print(f"  OOD ({method_label} < {args.confidence_threshold}%): {len(results['OOD'])} images")

    # Save results.
    os.makedirs(args.output_dir, exist_ok=True)

    if args.copy_images:
        for cls in class_names:
            cls_dir = os.path.join(args.output_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for p in results[cls]:
                shutil.copy2(p, cls_dir)

        ood_dir = os.path.join(args.output_dir, "OOD")
        os.makedirs(ood_dir, exist_ok=True)
        for item in results["OOD"]:
            shutil.copy2(item["path"], ood_dir)

        print(f"\nImages copied to {args.output_dir}/")

    # Save JSON summary.
    summary = {
        cls: len(results[cls]) if cls != "OOD" else len(results["OOD"])
        for cls in list(class_names) + ["OOD"]
    }
    summary_path = os.path.join(args.output_dir, "results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with trained classifier")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model (.pth)")
    parser.add_argument("--classes_path", type=str, required=True,
                        help="Path to class names file (one per line)")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to images for classification")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--ood_method", type=str, default="softmax",
                        choices=["softmax", "cosine"],
                        help="OOD detection method: 'softmax' or 'cosine'")
    parser.add_argument("--confidence_threshold", type=float, default=96.0,
                        help="For softmax: confidence %% (default 96). "
                             "For cosine: percentile of training similarities "
                             "(default 7; lower = more conservative).")
    parser.add_argument("--prototypes_path", type=str, default=None,
                        help="Path to prototypes .pt file (required for cosine OOD)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--copy_images", action="store_true",
                        help="Copy classified images into per-class folders")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())

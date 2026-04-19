"""Gradio web interface for the DNA image classification pipeline.

Provides an interactive UI for:
  1. Cropping individual DNA objects from microscopy images
  2. Labelling cropped images
  3. Training a classifier
  4. Running inference and reviewing OOD samples
  5. Iterative re-labelling and retraining

Launch::

    python app.py [--port 7860] [--share]
"""

import argparse
import gc
import glob
import os
import shutil
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime
from itertools import chain

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.model import Classifier, compute_prototypes, cosine_ood_scores, percentile_threshold
from src.data import (
    BalancedClassSampler,
    UnlabeledDataset,
    get_eval_transform,
    get_train_transform,
)
from src.preprocess import crop_objects_from_image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff")
CLASS_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
]

# Default training hyper-parameters.
EPOCHS = 13
BATCH_SIZE = 256
LR_MLP = 2.5e-05
LR_BACKBONE_LATE = 5.1e-06
LR_BACKBONE_MIDDLE = 4.1e-06
LR_BACKBONE_EARLY = 7.3e-07

# Transforms.
train_transform = get_train_transform(224)
eval_transform = get_eval_transform(224)

# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

MAX_AVAILABLE_GPUS = torch.cuda.device_count()


def get_gpu_choices():
    choices = ["CPU Only"]
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        choices.insert(0, f"Auto: Use All ({n} GPUs)")
        for i in range(n):
            choices.append(f"Use GPU {i}")
    return choices


GPU_CHOICES = get_gpu_choices()


def parse_gpu_config(selection_str):
    if not selection_str or "CPU" in selection_str:
        return torch.device("cpu"), False
    if "Auto" in selection_str:
        return torch.device("cuda"), True
    if "Use GPU" in selection_str:
        gpu_id = int(selection_str.split()[-1])
        if gpu_id >= MAX_AVAILABLE_GPUS:
            gr.Warning(f"GPU {gpu_id} unavailable, falling back to GPU 0.")
            gpu_id = 0
        return torch.device(f"cuda:{gpu_id}"), False
    return torch.device("cpu"), False


# ---------------------------------------------------------------------------
# Image display helpers
# ---------------------------------------------------------------------------

def create_bordered_image(original_path, temp_dir, label=None, class_order=None):
    """Return a copy of the image with a coloured border if labelled."""
    base, ext = os.path.splitext(os.path.basename(original_path))
    ts = int(datetime.now().timestamp())

    if label and class_order:
        try:
            idx = class_order.index(label)
            color = CLASS_COLORS[idx % len(CLASS_COLORS)]
            img = cv2.imread(original_path)
            if img is not None:
                img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=color)
                out = os.path.join(temp_dir, f"{base}__{label}__{ts}{ext}")
                cv2.imwrite(out, img)
                return out
        except (ValueError, IndexError):
            pass

    out = os.path.join(temp_dir, f"{base}__plain__{ts}{ext}")
    try:
        shutil.copy2(original_path, out)
        return out
    except Exception:
        return original_path


def refresh_gallery_with_borders(ood_data, labeled_data, class_order, temp_dir):
    path_to_label = {p: c for c, ps in labeled_data.items() for p in ps}
    if not ood_data:
        return gr.update(value=[])
    display = []
    for path, pred_class, conf in ood_data:
        bordered = create_bordered_image(path, temp_dir, path_to_label.get(path), class_order)
        display.append((bordered, f"{pred_class} ({conf:.1f}%)"))
    return gr.update(value=display)


# ---------------------------------------------------------------------------
# Crop previews
# ---------------------------------------------------------------------------

def load_full_images_from_upload(files):
    if not files:
        gr.Warning("No files uploaded.")
        return gr.update(choices=[], value=None), None
    filepaths = [f.name for f in files]
    filenames = [os.path.basename(p) for p in filepaths]
    return gr.update(choices=filenames, value=filenames[0]), filepaths


def update_crop_previews(selected_filename, all_paths, threshold, padding):
    if not selected_filename or not all_paths:
        return None, None
    image_path = next((p for p in all_paths if os.path.basename(p) == selected_filename), None)
    if not image_path:
        return None, None

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None, None

    from skimage import measure, morphology

    h, w = gray.shape
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    clean = morphology.remove_small_objects(binary.astype(bool), min_size=60)
    labelled = measure.label(clean)
    props = measure.regionprops(labelled)

    vis = cv2.cvtColor(clean.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    for prop in props:
        coords = prop.coords
        rows, cols = coords[:, 0], coords[:, 1]
        if np.any(rows == 0) or np.any(rows == h - 1) or np.any(cols == 0) or np.any(cols == w - 1):
            continue
        minr, minc, maxr, maxc = prop.bbox
        cv2.rectangle(
            vis,
            (max(0, minc - padding), max(0, minr - padding)),
            (min(w, maxc + padding), min(h, maxr + padding)),
            (0, 0, 255), 2,
        )
    return gray, vis


def run_cropping_on_selected(selected_filename, all_paths, threshold, padding, output_path):
    if not selected_filename or not all_paths:
        raise gr.Error("Please upload and select an image first.")
    image_path = next((p for p in all_paths if os.path.basename(p) == selected_filename), None)
    if not image_path:
        raise gr.Error("Image path not found.")

    if not output_path:
        output_path = tempfile.mkdtemp(prefix="cropped_")
    os.makedirs(output_path, exist_ok=True)

    saved = crop_objects_from_image(image_path, output_path, threshold, padding)
    total = len(glob.glob(os.path.join(output_path, "*.*")))
    status = f"Cropped {len(saved)} objects from '{selected_filename}' ({total} total in folder)."
    return output_path, status


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

def load_images_for_labeling(data_path, temp_dir):
    if not data_path or not os.path.isdir(data_path):
        gr.Warning("Cropped image folder is not ready.")
        return [], {}, [], [], "Labelling status", []
    all_imgs = sorted([
        p for p in glob.glob(os.path.join(data_path, "**", "*.*"), recursive=True)
        if p.lower().endswith(SUPPORTED_EXTENSIONS)
    ])
    if not all_imgs:
        gr.Warning(f"No images found in '{data_path}'.")
        return [], {}, [], [], "Labelling status", []
    display = [create_bordered_image(p, temp_dir) for p in all_imgs]
    gr.Info(f"Loaded {len(all_imgs)} images.")
    return all_imgs, all_imgs, {}, display, "Labelling status", []


def update_class_dropdowns(class_names_str):
    names = [n.strip() for n in class_names_str.split(",") if n.strip()]
    return names, gr.update(choices=names), gr.update(choices=names), gr.update(choices=names)


def toggle_label_on_click(evt: gr.SelectData, all_paths_data, selected_class,
                          labeled_data, class_order, temp_dir):
    if not selected_class:
        gr.Warning("Select a class first!")
        return labeled_data, "Labelling status", gr.update()

    is_ood = all_paths_data and isinstance(all_paths_data[0], (tuple, list))
    clicked_path = all_paths_data[evt.index][0] if is_ood else all_paths_data[evt.index]

    data = dict(labeled_data)
    for cls, paths in data.items():
        if cls != selected_class and clicked_path in paths:
            data[cls].remove(clicked_path)

    current = set(data.get(selected_class, []))
    if clicked_path in current:
        current.remove(clicked_path)
    else:
        current.add(clicked_path)
    data[selected_class] = sorted(current)

    # Status.
    total = sum(len(ps) for ps in data.values())
    status = "**Labelling status**\n"
    for c, ps in sorted(data.items()):
        if ps:
            status += f"- **{c}**: {len(ps)}\n"
    status += f"\n**Total: {total}**"

    # Rebuild gallery.
    label_map = {p: c for c, ps in data.items() for p in ps}
    display = []
    if is_ood:
        for path, pred_class, conf in all_paths_data:
            bordered = create_bordered_image(path, temp_dir, label_map.get(path), class_order)
            display.append((bordered, f"{pred_class} ({conf:.1f}%)"))
    else:
        for path in all_paths_data:
            display.append(create_bordered_image(path, temp_dir, label_map.get(path), class_order))

    return data, status, display


def create_tmp_dataset_folder(labeled_data, work_dir):
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    for class_name, paths in labeled_data.items():
        cls_dir = os.path.join(work_dir, class_name)
        os.makedirs(cls_dir, exist_ok=True)
        for p in paths:
            shutil.copy(p, cls_dir)
    return work_dir


# ---------------------------------------------------------------------------
# Training & Inference
# ---------------------------------------------------------------------------

def train_model(dataset_path, num_classes, gpu_str, progress=gr.Progress(track_tqdm=True)):
    device, use_dp = parse_gpu_config(gpu_str)
    dataset = ImageFolder(root=dataset_path, transform=train_transform)

    loader_kwargs = {"batch_size": BATCH_SIZE, "num_workers": min(8, os.cpu_count() or 1)}
    if num_classes > 1:
        loader_kwargs["sampler"] = BalancedClassSampler(dataset, num_classes=num_classes)
    else:
        loader_kwargs["shuffle"] = True
    dataloader = DataLoader(dataset, **loader_kwargs)

    model = Classifier(num_classes=num_classes).to(device)
    if use_dp and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    m = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = optim.AdamW([
        {"params": chain(m.backbone.conv1.parameters(), m.backbone.bn1.parameters(),
                         m.backbone.layer1.parameters()), "lr": LR_BACKBONE_EARLY},
        {"params": m.backbone.layer2.parameters(), "lr": LR_BACKBONE_MIDDLE},
        {"params": chain(m.backbone.layer3.parameters(), m.backbone.layer4.parameters()),
         "lr": LR_BACKBONE_LATE},
        {"params": m.mlp.parameters(), "lr": LR_MLP},
    ])
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        for imgs, lbls in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()

    save_dir = tempfile.mkdtemp()
    ts = f"{datetime.now():%Y%m%d_%H%M%S}"
    model_path = os.path.join(save_dir, f"model_{ts}.pth")
    torch.save(m.state_dict(), model_path)

    # Compute and save prototypes for cosine OOD.
    m.eval()
    eval_ds = ImageFolder(root=dataset_path, transform=eval_transform)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=min(8, os.cpu_count() or 1))
    prototypes, train_sim = compute_prototypes(m, eval_loader, num_classes, device)
    proto_path = os.path.join(save_dir, f"prototypes_{ts}.pt")
    torch.save({"prototypes": prototypes, "train_sim": train_sim}, proto_path)

    del model, dataloader, dataset, optimizer, criterion, eval_ds, eval_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return model_path, proto_path


def run_inference(model_path, class_names, data_root, confidence_threshold, gpu_str,
                  ood_method="softmax", proto_path=None,
                  progress=gr.Progress(track_tqdm=True)):
    if not data_root or not os.path.isdir(data_root):
        raise gr.Error("Please provide a valid image directory.")

    device, use_dp = parse_gpu_config(gpu_str)
    model = Classifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    if use_dp and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    # Load prototypes for cosine OOD.
    prototypes = None
    train_sim = None
    if ood_method == "cosine":
        if proto_path and os.path.exists(proto_path):
            data = torch.load(proto_path, map_location="cpu")
            if isinstance(data, dict):
                prototypes = data["prototypes"]
                train_sim = data.get("train_sim")
            else:
                prototypes = data  # legacy: bare tensor
        else:
            raise gr.Error("Cosine OOD requires prototypes. Train a model first.")

    paths = sorted([
        p for p in glob.glob(os.path.join(data_root, "**", "*.*"), recursive=True)
        if p.lower().endswith(SUPPORTED_EXTENSIONS)
    ])
    if not paths:
        return {}, [], "No images found.", ""

    ds = UnlabeledDataset(paths, transform=eval_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=min(8, os.cpu_count() or 1))

    results = {n: [] for n in class_names}
    results["OOD"] = []

    m = model.module if isinstance(model, nn.DataParallel) else model

    # Determine threshold.
    if ood_method == "cosine":
        # Slider value is interpreted as percentile (0-100).
        p = confidence_threshold
        if train_sim is not None:
            threshold = percentile_threshold(train_sim, p)
        else:
            threshold = p / 100.0  # fallback: use as raw fraction
        threshold_display = f"percentile {p:.0f} (sim={threshold:.3f})"
    else:
        threshold = confidence_threshold / 100.0
        threshold_display = f"{confidence_threshold:.0f}%"

    with torch.no_grad():
        for imgs, p_batch in tqdm(loader, desc="Inference"):
            imgs = imgs.to(device)
            if ood_method == "cosine":
                confs, preds = cosine_ood_scores(m, imgs, prototypes, device)
            else:
                probs = torch.softmax(model(imgs), dim=1)
                confs, preds = torch.max(probs, dim=1)
            for i in range(len(p_batch)):
                conf = confs[i].item()
                pred_name = class_names[preds[i].item()]
                if conf < threshold:
                    results["OOD"].append((p_batch[i], pred_name, conf * 100.0))
                else:
                    results[pred_name].append(p_batch[i])

    ood = results.get("OOD", [])
    method_label = "Cosine similarity" if ood_method == "cosine" else "Softmax confidence"
    summary = f"**Classification Results ({method_label})**\n" + "\n".join(
        f"- **{c}**: {len(ps)}" for c, ps in results.items()
    ) + f"\n\n(Threshold: {threshold_display})"
    status = f"Done. {len(paths)} images processed. OOD: {len(ood)}"

    del model, ds, loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return results, ood, status, summary


# ---------------------------------------------------------------------------
# Pipeline wrappers
# ---------------------------------------------------------------------------

def run_initial_training_and_eval(labeled_data, cropped_root, confidence, gpu,
                                  ood_method="softmax",
                                  progress=gr.Progress(track_tqdm=True)):
    if len(labeled_data) < 2:
        raise gr.Error("At least 2 classes are required for training.")
    tmp_dir = tempfile.mkdtemp()
    create_tmp_dataset_folder(labeled_data, tmp_dir)
    names = sorted(labeled_data.keys())
    model_path, proto_path = train_model(tmp_dir, len(names), gpu, progress)
    res, ood, status, summary = run_inference(
        model_path, names, cropped_root, confidence, gpu,
        ood_method=ood_method, proto_path=proto_path, progress=progress,
    )
    shutil.rmtree(tmp_dir)
    return (model_path, proto_path, res, ood, [], status,
            gr.update(choices=sorted(res.keys())), summary,
            gr.update(value=model_path, visible=True))


def run_retraining_and_eval(labeled_data, cropped_root, confidence, gpu,
                            ood_method="softmax",
                            progress=gr.Progress(track_tqdm=True)):
    if len(labeled_data) < 2:
        raise gr.Error("At least 2 classes are required for training.")
    tmp_dir = tempfile.mkdtemp()
    create_tmp_dataset_folder(labeled_data, tmp_dir)
    names = sorted(labeled_data.keys())
    model_path, proto_path = train_model(tmp_dir, len(names), gpu, progress)
    res, ood, _, summary = run_inference(
        model_path, names, cropped_root, confidence, gpu,
        ood_method=ood_method, proto_path=proto_path, progress=progress,
    )
    shutil.rmtree(tmp_dir)
    return (model_path, proto_path, res, ood, [], "Retraining complete.",
            gr.update(choices=sorted(res.keys())), summary,
            gr.update(value=model_path, visible=True))


def run_inference_with_loaded_model(model_file, proto_file, class_names_str,
                                    cropped_root, confidence, gpu,
                                    ood_method="softmax",
                                    progress=gr.Progress(track_tqdm=True)):
    if not model_file:
        raise gr.Error("Please upload a model (.pth) file.")
    if not class_names_str:
        raise gr.Error("Please enter class names (comma-separated, same order as training).")
    if ood_method == "cosine" and not proto_file:
        raise gr.Error("Cosine OOD requires a prototypes (.pt) file.")
    names = [n.strip() for n in class_names_str.split(",")]
    proto_path = proto_file.name if proto_file else None
    res, ood, status, summary = run_inference(
        model_file.name, names, cropped_root, confidence, gpu,
        ood_method=ood_method, proto_path=proto_path, progress=progress,
    )
    return res, ood, [], status, gr.update(choices=sorted(res.keys())), summary


def update_result_gallery(selected_class, results_dict):
    if not selected_class or not results_dict:
        return []
    items = results_dict.get(selected_class, [])
    if items and isinstance(items[0], (tuple, list)):
        return [(it[0], f"{it[1]} ({float(it[2]):.1f}%)") if len(it) == 3
                else (it[0], it[1]) if len(it) == 2 else it[0]
                for it in items]
    return items


def prepare_class_zip_download(class_name, results_dict):
    if not class_name or not results_dict:
        gr.Warning("Select a class to download.")
        return None
    paths = results_dict.get(class_name)
    if not paths:
        gr.Warning(f"No images in class '{class_name}'.")
        return None
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, "w") as zf:
        for p in paths:
            zf.write(p, os.path.basename(p))
    gr.Info(f"Prepared {len(paths)} images from '{class_name}' as ZIP.")
    return zip_path


def switch_start_mode(choice):
    return (gr.update(visible=(choice == "Train New Model")),
            gr.update(visible=(choice == "Load Existing Model")))


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="DNA Image Classification Pipeline") as demo:
        # --- State ---
        state_all_images = gr.State([])
        state_display_images = gr.State([])
        state_labeled_data = gr.State({})
        state_class_order = gr.State([])
        state_selected_indices = gr.State([])
        state_temp_image_dir = gr.State(tempfile.mkdtemp(prefix="bordered_"))
        state_ood_images = gr.State([])
        state_model_path = gr.State(None)
        state_proto_path = gr.State(None)
        state_results = gr.State({})
        state_full_image_paths = gr.State([])
        state_cropped_output_path = gr.State(None)

        gr.Markdown("# DNA Image Classification Pipeline")
        radio_mode = gr.Radio(
            ["Train New Model", "Load Existing Model"],
            label="Mode", value="Train New Model",
        )

        # ===== Train new model =====
        with gr.Column(visible=True) as col_train:
            with gr.Tabs() as tabs:
                # -- Step 0: Cropping --
                with gr.TabItem("Step 0: Image Cropping"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            file_uploader = gr.File(
                                label="Upload full-field images",
                                file_count="multiple", file_types=["image"], type="filepath",
                            )
                            dd_image_select = gr.Dropdown(label="Select image to preview", interactive=True)
                            slider_thresh = gr.Slider(label="Binary threshold", minimum=0, maximum=255, value=80, step=1)
                            slider_pad = gr.Slider(label="Padding (px)", minimum=0, maximum=50, value=3, step=1)
                            btn_crop = gr.Button("Crop Selected Image", variant="secondary")
                            md_crop_status = gr.Markdown("Crop status")
                        with gr.Column(scale=3):
                            with gr.Row():
                                img_orig = gr.Image(label="Original (Grayscale)")
                                img_binary = gr.Image(label="Bounding Box Preview")

                # -- Step 1: Labelling --
                with gr.TabItem("Step 1: Data Preparation & Labelling"):
                    gr.Markdown(
                        "1. Define class names below.\n"
                        "2. Click images in the gallery to assign / toggle labels.\n"
                    )
                    with gr.Row():
                        with gr.Column(scale=1):
                            default_gpu = GPU_CHOICES[0]
                            if "Use GPU 0" in GPU_CHOICES:
                                default_gpu = "Use GPU 0"
                            dd_gpu = gr.Dropdown(choices=GPU_CHOICES, value=default_gpu, label="GPU")
                            btn_load = gr.Button("Load Cropped Images", variant="primary")
                            txt_classes = gr.Textbox(label="Class names (comma-separated)", placeholder="ClassA, ClassB")
                            dd_label = gr.Dropdown(label="Class to assign", interactive=True)
                            md_label_status = gr.Markdown("Labelling status")
                        with gr.Column(scale=3):
                            gallery_label = gr.Gallery(
                                label="Images (click to label/unlabel)", show_label=False,
                                columns=8, height="600px", object_fit="contain", allow_preview=False,
                            )

                # -- Step 2: Initial training --
                with gr.TabItem("Step 2: Train & Infer"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            btn_train = gr.Button("Train + Infer", variant="primary")
                            radio_ood = gr.Radio(
                                ["softmax", "cosine"],
                                value="softmax",
                                label="OOD Detection Method",
                            )
                            slider_conf = gr.Slider(minimum=0, maximum=100, value=96, step=1, label="OOD Threshold (%)")
                            md_train_status = gr.Markdown("Training status")
                        with gr.Column(scale=2):
                            file_model_dl = gr.File(label="Download trained model", visible=False)

                # -- Step 3: OOD labelling --
                with gr.TabItem("Step 3: OOD Labelling"):
                    gr.Markdown("Label OOD images the same way as Step 1.")
                    with gr.Row():
                        with gr.Column(scale=1):
                            dd_ood_label = gr.Dropdown(label="Class to assign", interactive=True)
                            md_ood_status = gr.Markdown("Labelling status")
                        with gr.Column(scale=3):
                            gallery_ood = gr.Gallery(
                                label="OOD images (click to label/unlabel)", show_label=False,
                                columns=8, height="600px", object_fit="contain", allow_preview=False,
                            )

                # -- Step 4: Retraining --
                with gr.TabItem("Step 4: Retrain"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            btn_retrain = gr.Button("Retrain + Final Infer", variant="primary")
                            radio_ood2 = gr.Radio(
                                ["softmax", "cosine"],
                                value="softmax",
                                label="OOD Detection Method",
                            )
                            slider_conf2 = gr.Slider(minimum=0, maximum=100, value=96, step=1, label="OOD Threshold (%)")
                            md_retrain_status = gr.Markdown("Retraining status")

                # -- Step 5: Results --
                with gr.TabItem("Step 5: Results"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            dd_result_class = gr.Dropdown(label="Select class to view")
                            md_summary = gr.Markdown("Classification summary")
                            btn_zip = gr.Button("Download class images (ZIP)", variant="secondary")
                            file_zip = gr.File(label="ZIP download")
                        with gr.Column(scale=3):
                            gallery_results = gr.Gallery(
                                label="Classification results", show_label=False,
                                columns=8, height="600px", object_fit="contain",
                            )

        # ===== Load existing model =====
        with gr.Column(visible=False) as col_load:
            gr.Markdown("## Inference with Existing Model")
            with gr.Row():
                with gr.Column():
                    file_model_upload = gr.File(label="Upload model (.pth)", type="filepath")
                    file_proto_upload = gr.File(label="Upload prototypes (.pt, for cosine OOD)", type="filepath")
                    txt_model_classes = gr.Textbox(
                        label="Class names (same order as training, comma-separated)",
                        placeholder="ClassA,ClassB",
                    )
                    default_gpu_load = GPU_CHOICES[0]
                    if "Use GPU 0" in GPU_CHOICES:
                        default_gpu_load = "Use GPU 0"
                    dd_gpu_load = gr.Dropdown(choices=GPU_CHOICES, value=default_gpu_load, label="GPU")
                    radio_ood_load = gr.Radio(
                        ["softmax", "cosine"],
                        value="softmax",
                        label="OOD Detection Method",
                    )
                    slider_conf_load = gr.Slider(minimum=0, maximum=100, value=96, step=1, label="OOD Threshold (%)")
                    btn_infer_loaded = gr.Button("Run Inference", variant="primary")
                    md_infer_status = gr.Markdown("Inference status")

        # --- Event wiring ---
        radio_mode.change(switch_start_mode, radio_mode, [col_train, col_load])

        file_uploader.upload(load_full_images_from_upload, file_uploader, [dd_image_select, state_full_image_paths])

        preview_inputs = [dd_image_select, state_full_image_paths, slider_thresh, slider_pad]
        preview_outputs = [img_orig, img_binary]
        dd_image_select.change(update_crop_previews, preview_inputs, preview_outputs)
        slider_thresh.release(update_crop_previews, preview_inputs, preview_outputs)
        slider_pad.release(update_crop_previews, preview_inputs, preview_outputs)

        btn_crop.click(
            run_cropping_on_selected,
            [dd_image_select, state_full_image_paths, slider_thresh, slider_pad, state_cropped_output_path],
            [state_cropped_output_path, md_crop_status],
        ).then(
            load_images_for_labeling,
            [state_cropped_output_path, state_temp_image_dir],
            [state_all_images, state_display_images, state_labeled_data, gallery_label, md_label_status, state_selected_indices],
        ).then(
            lambda: gr.update(selected="Step 1: Data Preparation & Labelling"),
            outputs=tabs,
        )

        btn_load.click(
            load_images_for_labeling,
            [state_cropped_output_path, state_temp_image_dir],
            [state_all_images, state_display_images, state_labeled_data, gallery_label, md_label_status, state_selected_indices],
        )

        txt_classes.change(
            update_class_dropdowns, txt_classes,
            [state_class_order, dd_label, dd_ood_label, dd_result_class],
        )

        gallery_label.select(
            toggle_label_on_click,
            [state_all_images, dd_label, state_labeled_data, state_class_order, state_temp_image_dir],
            [state_labeled_data, md_label_status, gallery_label],
        )

        train_outputs = [
            state_model_path, state_proto_path, state_results, state_ood_images, gallery_ood,
            md_train_status, dd_result_class, md_summary, file_model_dl,
        ]
        btn_train.click(
            run_initial_training_and_eval,
            [state_labeled_data, state_cropped_output_path, slider_conf, dd_gpu, radio_ood],
            train_outputs,
        ).then(
            refresh_gallery_with_borders,
            [state_ood_images, state_labeled_data, state_class_order, state_temp_image_dir],
            [gallery_ood],
        )

        gallery_ood.select(
            toggle_label_on_click,
            [state_ood_images, dd_ood_label, state_labeled_data, state_class_order, state_temp_image_dir],
            [state_labeled_data, md_ood_status, gallery_ood],
        )

        retrain_outputs = [
            state_model_path, state_proto_path, state_results, state_ood_images, gallery_ood,
            md_retrain_status, dd_result_class, md_summary, file_model_dl,
        ]
        btn_retrain.click(
            run_retraining_and_eval,
            [state_labeled_data, state_cropped_output_path, slider_conf2, dd_gpu, radio_ood2],
            retrain_outputs,
        )

        dd_result_class.change(update_result_gallery, [dd_result_class, state_results], [gallery_results])
        btn_zip.click(prepare_class_zip_download, [dd_result_class, state_results], [file_zip])

        infer_outputs = [state_results, state_ood_images, gallery_ood, md_infer_status, dd_result_class, md_summary]
        btn_infer_loaded.click(
            run_inference_with_loaded_model,
            [file_model_upload, file_proto_upload, txt_model_classes,
             state_cropped_output_path, slider_conf_load, dd_gpu_load, radio_ood_load],
            infer_outputs,
        ).then(
            refresh_gallery_with_borders,
            [state_ood_images, state_labeled_data, state_class_order, state_temp_image_dir],
            [gallery_ood],
        ).then(
            lambda: gr.update(selected="Step 5: Results"),
            outputs=tabs,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_api=False,
    )

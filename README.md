# DNA Image Classification Pipeline

A deep learning pipeline for classifying DNA morphological structures from microscopy images. The system uses a ResNet-50 backbone with a two-layer MLP head, fine-tuned with layer-wise learning rates and balanced class sampling.

## Features

- **Object cropping**: Automatic segmentation of individual DNA objects from full-field microscopy images using binary thresholding and connected-component analysis
- **Aspect-ratio-preserving resize**: Custom `ResizeWithPad` transform that pads with ImageNet mean colour
- **Balanced training**: Oversampling of minority classes via `BalancedClassSampler`
- **Layer-wise learning rates**: Differential learning rates for early, middle, and late backbone layers
- **OOD detection**: Confidence-based out-of-distribution filtering at inference time
- **Interactive UI**: Gradio web interface for the full labelling → training → inference loop

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training (CLI)

Prepare your data in [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) format:

```
data/train/
├── ClassA/
│   ├── img001.png
│   └── ...
├── ClassB/
│   └── ...
```

Then run:

```bash
python train.py --data_root ./data/train --epochs 13 --gpu 0
```

### Inference (CLI)

```bash
python inference.py \
    --model_path ./checkpoints/model.pth \
    --classes_path ./checkpoints/classes.txt \
    --data_root ./data/test \
    --confidence_threshold 96 \
    --copy_images
```

### Gradio Web Interface

```bash
python app.py --port 7860
```

## Model Architecture

| Component | Details |
|-----------|---------|
| Backbone | ResNet-50 (ImageNet V2 pretrained) |
| Head | Linear(2048, 4096) → ReLU → Dropout(0.7) → Linear(4096, *C*) |
| Optimizer | AdamW with layer-wise LR |
| Loss | Cross-entropy |
| Input size | 224 × 224 (aspect-ratio preserving pad) |

### Layer-wise Learning Rates

| Layer group | Learning rate |
|-------------|--------------|
| conv1, bn1, layer1 (early) | 7.3 × 10⁻⁷ |
| layer2 (middle) | 4.1 × 10⁻⁶ |
| layer3, layer4 (late) | 5.1 × 10⁻⁶ |
| MLP head | 2.5 × 10⁻⁵ |

## Project Structure

```
├── train.py            # CLI training script
├── inference.py        # CLI inference script
├── app.py              # Gradio web demo
├── requirements.txt
└── src/
    ├── model.py        # Classifier and MLPHead
    ├── data.py         # Transforms, datasets, samplers
    └── preprocess.py   # Microscopy image cropping
```

## Citation

```bibtex
@article{TODO,
  title={TODO},
  author={TODO},
  year={2025}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

"""Object cropping from microscopy images.

Segments individual DNA objects from full-field microscopy images using
binary thresholding, morphological cleaning, and connected-component
analysis.  Objects touching the image border are excluded.
"""

import os
from typing import List, Tuple

import cv2
import numpy as np
from skimage import measure, morphology


def crop_objects_from_image(
    image_path: str,
    output_dir: str,
    threshold: int = 80,
    padding: int = 3,
    min_object_size: int = 60,
) -> List[str]:
    """Crop individual objects from a microscopy image.

    Args:
        image_path: Path to the input image.
        output_dir: Directory where cropped images will be saved.
        threshold: Binarisation threshold (0–255).
        padding: Pixel padding around each bounding box.
        min_object_size: Minimum object area in pixels to keep.

    Returns:
        List of paths to saved cropped images.
    """
    os.makedirs(output_dir, exist_ok=True)

    original_bgr = cv2.imread(image_path)
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_bgr is None or gray is None:
        return []

    img_h, img_w = gray.shape

    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    binary_clean = morphology.remove_small_objects(
        binary.astype(bool), min_size=min_object_size,
    )

    labelled = measure.label(binary_clean, connectivity=1)
    props = measure.regionprops(labelled)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    saved_paths: List[str] = []

    for i, prop in enumerate(props):
        coords = prop.coords
        rows, cols = coords[:, 0], coords[:, 1]

        # Skip objects touching the image border.
        touches_border = (
            np.any(rows == 0)
            or np.any(rows == img_h - 1)
            or np.any(cols == 0)
            or np.any(cols == img_w - 1)
        )
        if touches_border:
            continue

        minr, minc, maxr, maxc = prop.bbox
        p_minr = max(0, minr - padding)
        p_minc = max(0, minc - padding)
        p_maxr = min(img_h, maxr + padding)
        p_maxc = min(img_w, maxc + padding)

        out_path = os.path.join(output_dir, f"{base_name}_object_{i + 1}.png")
        cv2.imwrite(out_path, original_bgr[p_minr:p_maxr, p_minc:p_maxc])
        saved_paths.append(out_path)

    return saved_paths

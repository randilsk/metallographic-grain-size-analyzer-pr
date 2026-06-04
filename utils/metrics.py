# utils/metrics.py

import cv2
import numpy as np
import os
from config import SEGMENTED_DIR

def compare_with_ground_truth(skeleton, image_filename):
    """
    Compare pipeline skeleton output with manual segmentation.
    Returns precision, recall, f1 score.
    """

    name = os.path.splitext(image_filename)[0]

    # Search for matching seg file with any extension
    seg_path = None
    for ext in [".png", ".jpg", ".bmp", ".tif"]:
        candidate = os.path.join(SEGMENTED_DIR, f"{name}_seg{ext}")
        if os.path.exists(candidate):
            seg_path = candidate
            break

    if seg_path is None:
        print(f"No ground truth found for {image_filename}")
        print(f"  Looking for: {name}_seg.*")
        files = os.listdir(SEGMENTED_DIR)
        matching = [f for f in files if name in f]
        if matching:
            print(f"  Found similar: {matching}")
        return None

    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if seg is None:
        print(f"Could not load: {seg_path}")
        return None

    # Resize if needed
    if seg.shape != skeleton.shape:
        seg = cv2.resize(seg, (skeleton.shape[1], skeleton.shape[0]))

    # Ground truth: white = grain interior, black = boundary
    # Invert so boundaries become white to match our skeleton
    seg = cv2.bitwise_not(seg)

    # Dilate skeleton to match ground truth boundary thickness (~10px wide in GT)
    # 11×11 ellipse provides ~5px radius coverage to properly overlap GT boundaries
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skeleton_thick = cv2.dilate(skeleton, dilate_kernel, iterations=1)

    # Binarize both — use skeleton_thick for pred
    _, gt   = cv2.threshold(seg, 127, 1, cv2.THRESH_BINARY)
    _, pred = cv2.threshold(skeleton_thick, 127, 1, cv2.THRESH_BINARY)

    # Calculate metrics
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\nMetrics for {image_filename}:")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  F1 Score  : {f1:.3f}")

    return precision, recall, f1
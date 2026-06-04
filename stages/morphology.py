# stages/morphology.py

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.util import invert
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MORPH_KERNEL_SIZE, CLOSE_KERNEL_SIZE,
                    MIN_AREA_THRESHOLD, PRUNE_LENGTH, MIN_SKELETON_CC)

def apply_morphology(binary):
    """
    Stage 3: Clean noise, thin boundaries to 1px skeleton, prune short branches,
    and remove small isolated skeleton fragments.
    Returns: cleaned, skeleton
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_KERNEL_SIZE)

    # Step 1 - Opening: removes small isolated noise pixels
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 2 - Remove small white regions (interior speckle from scratches)
    bool_img = cleaned.astype(bool)
    bool_img = remove_small_objects(bool_img, max_size=MIN_AREA_THRESHOLD)
    cleaned = (bool_img * 255).astype(np.uint8)

    # Step 3 - Closing with larger kernel: connects small gaps in boundaries
    # Using 5x5 kernel instead of 3x3 to better bridge fragmented boundaries
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    # Step 4 - Skeletonize: thin boundaries down to 1 pixel wide
    bool_img = cleaned.astype(bool)
    skeleton = skeletonize(bool_img)
    skeleton = (skeleton * 255).astype(np.uint8)

    # Step 5 - Prune short dangling branches (aggressively)
    skeleton = prune_skeleton(skeleton, PRUNE_LENGTH)

    # Step 6 - Remove small isolated skeleton fragments
    # Real grain boundaries form large connected networks.
    # Small isolated fragments are noise artifacts.
    skeleton = remove_small_skeleton_components(skeleton, MIN_SKELETON_CC)
    
    return cleaned, skeleton


def prune_skeleton(skeleton, min_length):
    """Remove short dangling branches from skeleton."""
    pruned = skeleton.copy()

    for _ in range(min_length):
        # Find endpoint pixels (pixels with only 1 neighbor)
        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(
            (pruned > 0).astype(np.uint8), -1, kernel
        )
        # An endpoint has itself + exactly 1 neighbor = count of 2
        endpoints = (pruned > 0) & (neighbor_count == 2)
        pruned[endpoints] = 0

    return pruned


def remove_small_skeleton_components(skeleton, min_size):
    """
    Remove connected components from the skeleton that are smaller
    than min_size pixels. Real grain boundaries form large connected
    networks; small isolated fragments are noise.
    """
    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skeleton, connectivity=8
    )
    
    # Create output, keeping only large components
    result = np.zeros_like(skeleton)
    for i in range(1, num_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            result[labels == i] = 255
    
    return result
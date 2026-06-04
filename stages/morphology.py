# stages/morphology.py

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from skimage.util import invert
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MORPH_KERNEL_SIZE, MIN_AREA_THRESHOLD, PRUNE_LENGTH

def apply_morphology(binary):
    """
    Stage 3: Clean noise, thin boundaries to 1px skeleton, prune short branches.
    Returns: cleaned, skeleton
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)

    # Step 1 - Opening: removes small isolated noise pixels
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Step 2 - Remove small white regions (interior speckle from scratches)
    bool_img = cleaned.astype(bool)
    bool_img = remove_small_objects(bool_img, max_size=MIN_AREA_THRESHOLD)
    cleaned = (bool_img * 255).astype(np.uint8)

    # Step 3 - Closing: connects small gaps in boundaries
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Step 4 - Skeletonize: thin boundaries down to 1 pixel wide
    bool_img = cleaned.astype(bool)
    skeleton = skeletonize(bool_img)
    skeleton = (skeleton * 255).astype(np.uint8)

    # Step 5 - Prune short dangling branches
    skeleton = prune_skeleton(skeleton, PRUNE_LENGTH)
    
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
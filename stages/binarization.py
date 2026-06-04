# stages/binarization.py

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C, CANNY_LOW, CANNY_HIGH

def binarize(enhanced):
    """
    Stage 2: Fuse adaptive threshold + Canny edges.
    Adaptive catches broad boundary regions.
    Canny catches subtle gradient-based boundaries.
    Combined = better recall without killing precision.
    """

    # Method 1 — Adaptive threshold (what we had before)
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    # Method 2 — Canny edge detection
    # Canny finds boundaries based on intensity gradients
    # threshold1=low, threshold2=high — edges between are kept
    # if connected to strong edges
    canny = cv2.Canny(enhanced, threshold1=CANNY_LOW, threshold2=CANNY_HIGH)

    # Fuse both — OR operation means a pixel is boundary
    # if EITHER method detected it
    fused = cv2.bitwise_or(adaptive, canny)

    return fused
# stages/binarization.py

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
                    CANNY_LOW, CANNY_HIGH, CANNY_DILATION_SIZE)

def binarize(enhanced):
    """
    Stage 2: Multi-method binarization with morphological gradient validation.
    
    Strategy:
    1. Adaptive threshold — catches boundary regions based on local mean
    2. Canny edges — precise gradient-based boundaries
    3. Morphological gradient — local intensity range detector
    
    Fusion: Keep adaptive pixels that are validated by EITHER Canny proximity
    OR strong morphological gradient. This catches real boundaries (which have
    strong local contrast) while filtering interior noise (which has weak contrast).
    """

    # Method 1 — Adaptive threshold: catches broad boundary regions
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    # Method 2 — Canny edge detection: precise gradient-based boundaries
    canny = cv2.Canny(enhanced, threshold1=CANNY_LOW, threshold2=CANNY_HIGH)

    # Method 3 — Morphological gradient: detects regions with high local contrast
    # Real grain boundaries have high local intensity variation;
    # interior noise/scratches have lower variation
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, morph_kernel)
    
    # Threshold the gradient to keep only high-contrast regions
    grad_thresh = np.percentile(gradient[gradient > 0], 50) if np.any(gradient > 0) else 10
    _, grad_mask = cv2.threshold(gradient, grad_thresh, 255, cv2.THRESH_BINARY)

    # Create Canny validation zone
    validate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CANNY_DILATION_SIZE, CANNY_DILATION_SIZE)
    )
    canny_zone = cv2.dilate(canny, validate_kernel, iterations=1)

    # Combine validation signals: pixel passes if near Canny OR high gradient
    validation_mask = cv2.bitwise_or(canny_zone, grad_mask)

    # Keep adaptive pixels only where validated
    fused = cv2.bitwise_and(adaptive, validation_mask)

    # Also include raw Canny edges (they are already precise)
    fused = cv2.bitwise_or(fused, canny)

    return fused
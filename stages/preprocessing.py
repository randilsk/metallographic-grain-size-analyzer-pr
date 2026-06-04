# stages/preprocessing.py

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (GAUSSIAN_KERNEL, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE,
                    NLM_H, NLM_TEMPLATE_WIN, NLM_SEARCH_WIN)

def preprocess(image_path):
    """
    Stage 1: Load image, convert to grayscale,
    denoise with Non-Local Means, sharpen with unsharp masking,
    enhance contrast with CLAHE.
    Returns: original_bgr, grayscale, denoised, enhanced
    """

    # Step 1 - Load image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Step 2 - Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Step 3 - Non-Local Means Denoising
    # NLM averages similar patches across the image, so it smooths out
    # repetitive textures (scratches, grain interior noise) while preserving
    # unique structures (grain boundaries). Far superior to bilateral for
    # this use case because scratches are repetitive patterns.
    denoised = cv2.fastNlMeansDenoising(
        gray,
        h=NLM_H,
        templateWindowSize=NLM_TEMPLATE_WIN,
        searchWindowSize=NLM_SEARCH_WIN
    )

    # Step 4 - Unsharp masking: sharpens boundary edges
    # Using a larger sigma (via kernel size) to be more selective —
    # sharpens large-scale boundary edges rather than fine scratches
    blur = cv2.GaussianBlur(denoised, GAUSSIAN_KERNEL, 0)
    unsharp = cv2.addWeighted(denoised, 1.3, blur, -0.3, 0)

    # Step 5 - CLAHE contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE
    )
    enhanced = clahe.apply(unsharp)

    return original, gray, denoised, enhanced
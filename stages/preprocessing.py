# stages/preprocessing.py

import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GAUSSIAN_KERNEL, CLAHE_CLIP_LIMIT, CLAHE_TILE_SIZE

def preprocess(image_path):
    """
    Stage 1: Load image, convert to grayscale,
    denoise with Gaussian blur, enhance contrast with CLAHE.
    Returns: original_bgr, grayscale, denoised, enhanced
    """

    # Step 1 - Load image
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Step 2 - Convert to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Step 3 - Bilateral filter: smooths texture, keeps edges sharp
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 4 - CLAHE contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE
    )
    enhanced = clahe.apply(denoised)

    return original, gray, denoised, enhanced
# stages/binarization.py

import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C

def binarize(enhanced):
    """
    Stage 2: Convert enhanced grayscale image to binary.
    Grain boundaries become white (255), grain interiors black (0).
    Returns: binary image
    """

    # Adaptive thresholding - handles uneven lighting
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,   # INV = boundaries become WHITE
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )

    return binary
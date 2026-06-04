# utils/ground_truth_analysis.py

import cv2
import numpy as np
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEGMENTED_DIR, MICRONS_PER_PIXEL, NUM_TEST_LINES


def compute_gt_skeleton(seg_path):
    """
    Convert ground truth filled-region image into a boundary skeleton
    so we can run the same ASTM analysis on it.
    Ground truth: white = grain interior, black = boundaries.
    We extract the boundary edges from it.
    """
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if seg is None:
        return None

    # Threshold to clean binary
    _, binary = cv2.threshold(seg, 127, 255, cv2.THRESH_BINARY)

    # Extract boundaries by finding edges between grain regions
    # Canny on the ground truth gives us the boundary lines
    boundaries = cv2.Canny(binary, 10, 50)

    # Dilate slightly to ensure connected boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundaries = cv2.dilate(boundaries, kernel, iterations=1)

    return boundaries, seg


def analyze_ground_truth(image_filename):
    """
    Run ASTM intercept analysis on the ground truth segmentation.
    Returns G number from ground truth.
    """
    name = os.path.splitext(image_filename)[0]
    seg_path = os.path.join(SEGMENTED_DIR, f"{name}_seg.jpg")

    if not os.path.exists(seg_path):
        print(f"  No ground truth found for {image_filename}")
        return None

    result = compute_gt_skeleton(seg_path)
    if result is None:
        return None

    gt_skeleton, seg = result
    h, w = gt_skeleton.shape

    # Run same ASTM intercept counting on ground truth skeleton
    test_lines = []
    for i in range(1, NUM_TEST_LINES + 1):
        y = int(h * i / (NUM_TEST_LINES + 1))
        test_lines.append(("H", y))
    for i in range(1, NUM_TEST_LINES + 1):
        x = int(w * i / (NUM_TEST_LINES + 1))
        test_lines.append(("V", x))

    total_length_mm     = 0
    total_intersections = 0

    for line_type, pos in test_lines:
        if line_type == "H":
            line_pixels    = gt_skeleton[pos, :]
            line_length_px = w
        else:
            line_pixels    = gt_skeleton[:, pos]
            line_length_px = h

        line_length_mm = line_length_px * MICRONS_PER_PIXEL / 1000

        intersections = 0
        in_boundary   = False
        for px in line_pixels:
            if px > 127 and not in_boundary:
                intersections += 1
                in_boundary = True
            elif px <= 127:
                in_boundary = False

        total_length_mm     += line_length_mm
        total_intersections += intersections

    if total_intersections == 0:
        return None

    mean_intercept_mm = total_length_mm / total_intersections
    G = (-6.6457 * math.log10(mean_intercept_mm)) - 3.298

    return G
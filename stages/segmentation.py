# stages/segmentation.py

import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GAP_CLOSE_DISTANCE

def segment(skeleton, enhanced):
    """
    Stage 4: Close gaps in skeleton boundaries, then use watershed
    to label each grain as a separate region.
    Returns: closed_skeleton, labels (each grain has unique integer label)
    """

    # Step 1 - Dilate skeleton slightly to close small gaps
    gap_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (GAP_CLOSE_DISTANCE, GAP_CLOSE_DISTANCE)
    )
    closed = cv2.dilate(skeleton, gap_kernel, iterations=1)
    closed = cv2.erode(closed, gap_kernel, iterations=1)

    # Step 2 - Fill enclosed grain regions to find grain centers (markers)
    # Invert so grain interiors are white, boundaries are black
    inverted = cv2.bitwise_not(closed)

    # Step 3 - Distance transform: bright at grain centers, dark at boundaries
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
    dist = dist.astype(np.uint8)

    # Step 4 - Threshold distance map to get definite grain center markers
    _, markers_bin = cv2.threshold(dist, 0.3 * dist.max(), 255, cv2.THRESH_BINARY)
    markers_bin = markers_bin.astype(np.uint8)

    # Step 5 - Label connected marker regions
    num_labels, markers = cv2.connectedComponents(markers_bin)

    # Step 6 - Watershed using markers
    # Watershed needs a 3-channel image
    img_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    markers = markers.astype(np.int32)
    cv2.watershed(img_3ch, markers)

    return closed, markers
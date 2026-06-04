# utils/ground_truth_analysis.py

import cv2
import numpy as np
import math
import os
import sys
# If skimage isn't in your env, you can use cv2.ximgproc.thinning instead
from skimage.morphology import skeletonize 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SEGMENTED_DIR, MICRONS_PER_PIXEL, NUM_TEST_LINES

def compute_gt_skeleton(seg_path):
    """
    Convert ground truth filled-region image into a 1-pixel wide boundary skeleton.
    """
    seg = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    if seg is None:
        return None

    _, binary = cv2.threshold(seg, 127, 255, cv2.THRESH_BINARY)

    # 1. Get boundaries (Canny or simple morphological gradient)
    boundaries = cv2.Canny(binary, 10, 50)

    # 2. Convert to boolean for skimage, then skeletonize to strictly 1-pixel width
    # This replaces the dilation step which ruins junction topology
    bool_boundaries = boundaries > 0
    skeleton = skeletonize(bool_boundaries)
    
    # Convert back to uint8 for OpenCV compatibility
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    return skeleton_uint8, seg

def get_crossing_number(skel_img, x, y):
    """
    Calculates the Rutovitz crossing number for a pixel at (x, y) 
    using its 3x3 neighborhood.
    """
    h, w = skel_img.shape
    if x == 0 or y == 0 or x == w - 1 or y == h - 1:
        return 2 # Ignore image borders for junctions

    # Extract 3x3 neighborhood and threshold to binary 0/1
    p = skel_img[y-1:y+2, x-1:x+2] > 127
    
    # Sequence of 8 neighbors going clockwise around the center pixel
    # P9 is P1 to complete the circle
    neighbors = [
        p[0,1], p[0,2], p[1,2], p[2,2], 
        p[2,1], p[2,0], p[1,0], p[0,0], p[0,1]
    ]
    
    crossings = 0
    for i in range(8):
        crossings += abs(int(neighbors[i]) - int(neighbors[i+1]))
        
    return crossings // 2

def analyze_ground_truth(image_filename):
    name = os.path.splitext(image_filename)[0]
    seg_path = os.path.join(SEGMENTED_DIR, f"{name}_seg.jpg")

    if not os.path.exists(seg_path):
        return None

    result = compute_gt_skeleton(seg_path)
    if result is None:
        return None

    gt_skeleton, seg = result
    h, w = gt_skeleton.shape

    test_lines = []
    for i in range(1, NUM_TEST_LINES + 1):
        y = int(h * i / (NUM_TEST_LINES + 1))
        test_lines.append(("H", y))
    for i in range(1, NUM_TEST_LINES + 1):
        x = int(w * i / (NUM_TEST_LINES + 1))
        test_lines.append(("V", x))

    total_length_mm = 0
    total_intersections = 0.0 # Float to handle 1.5 weights

    for line_type, pos in test_lines:
        line_length_px = w if line_type == "H" else h
        line_length_mm = line_length_px * MICRONS_PER_PIXEL / 1000
        
        total_length_mm += line_length_mm

        in_boundary = False
        
        # Iterate with an index so we know spatial coordinates for the neighborhood matrix
        for i in range(line_length_px):
            x = i if line_type == "H" else pos
            y = pos if line_type == "H" else i
            
            px_val = gt_skeleton[y, x]

            if px_val > 127 and not in_boundary:
                # We hit a boundary! Check topology before scoring.
                cn = get_crossing_number(gt_skeleton, x, y)
                
                if cn >= 3:
                    total_intersections += 1.5 # Triple/Quadruple junction
                else:
                    total_intersections += 1.0 # Standard continuous edge
                    
                in_boundary = True
            elif px_val <= 127:
                in_boundary = False

    if total_intersections == 0:
        return None

    mean_intercept_mm = total_length_mm / total_intersections
    G = (-6.6457 * math.log10(mean_intercept_mm)) - 3.298

    return G
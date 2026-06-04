# stages/analysis.py

import cv2
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MICRONS_PER_PIXEL, NUM_TEST_LINES

def get_crossing_number(skel_img, x, y):
    """
    Calculates the Rutovitz crossing number for a pixel at (x, y) 
    using its 3x3 neighborhood.
    """
    h, w = skel_img.shape
    # Ignore image borders to prevent index out-of-bounds
    if x == 0 or y == 0 or x == w - 1 or y == h - 1:
        return 2 

    # Extract 3x3 neighborhood and threshold to binary boolean
    p = skel_img[y-1:y+2, x-1:x+2] > 127
    
    # Sequence of 8 neighbors going clockwise around the center pixel
    neighbors = [
        p[0,1], p[0,2], p[1,2], p[2,2], 
        p[2,1], p[2,0], p[1,0], p[0,0], p[0,1]
    ]
    
    crossings = 0
    for i in range(8):
        crossings += abs(int(neighbors[i]) - int(neighbors[i+1]))
        
    return crossings // 2

def analyze(skeleton, image_filename):
    """
    Stage 5: ASTM Analysis
    Executes the standard intercept method with topological weighting
    for triple junctions (ASTM E112 compliance).
    """
    h, w = skeleton.shape

    # Generate uniform test lines
    test_lines = []
    for i in range(1, NUM_TEST_LINES + 1):
        y = int(h * i / (NUM_TEST_LINES + 1))
        test_lines.append(("H", y))
    for i in range(1, NUM_TEST_LINES + 1):
        x = int(w * i / (NUM_TEST_LINES + 1))
        test_lines.append(("V", x))

    total_length_mm = 0.0
    total_intersections = 0.0  # Float to handle 1.5 weights

    for line_type, pos in test_lines:
        line_length_px = w if line_type == "H" else h
        line_length_mm = line_length_px * MICRONS_PER_PIXEL / 1000
        total_length_mm += line_length_mm

        in_boundary = False
        
        # Raycast along the line
        for i in range(line_length_px):
            x = i if line_type == "H" else pos
            y = pos if line_type == "H" else i
            
            px_val = skeleton[y, x]

            if px_val > 127 and not in_boundary:
                # Boundary detected, perform topological matrix check
                cn = get_crossing_number(skeleton, x, y)
                
                if cn >= 3:
                    total_intersections += 1.5  # Junction
                else:
                    total_intersections += 1.0  # Standard edge
                    
                in_boundary = True
            elif px_val <= 127:
                in_boundary = False

    if total_intersections == 0:
        print(f"Warning: No intersections found for {image_filename}")
        return None

    # Calculate final ASTM metrics
    mean_intercept_mm = total_length_mm / total_intersections
    
    # Using the corrected negative coefficient formula
    G = (-6.6457 * math.log10(mean_intercept_mm)) - 3.298

    return {
        "G": G,
        "mean_intercept_mm": mean_intercept_mm,
        "total_intersections": total_intersections,
        "total_test_length_mm": total_length_mm
    }
# stages/analysis.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MICRONS_PER_PIXEL, NUM_TEST_LINES

def analyze(skeleton, image_filename="image"):
    """
    Stage 5: ASTM E112 intercept method.
    Draws test lines, counts boundary intersections,
    calculates mean intercept length and ASTM grain size number.
    """

    h, w = skeleton.shape
    results = []

    # ── Generate test lines ──────────────────────────────
    # Horizontal, vertical, and diagonal lines evenly spaced
    test_lines = []

    # Horizontal lines
    for i in range(1, NUM_TEST_LINES + 1):
        y = int(h * i / (NUM_TEST_LINES + 1))
        test_lines.append(("H", y))

    # Vertical lines
    for i in range(1, NUM_TEST_LINES + 1):
        x = int(w * i / (NUM_TEST_LINES + 1))
        test_lines.append(("V", x))

    total_length_mm   = 0
    total_intersections = 0

    # ── Count intersections ──────────────────────────────
    for line_type, pos in test_lines:

        if line_type == "H":
            # Extract row of pixels along this horizontal line
            line_pixels = skeleton[pos, :]
            line_length_px = w
        else:
            # Extract column of pixels along this vertical line
            line_pixels = skeleton[:, pos]
            line_length_px = h

        # Convert length to mm
        line_length_mm = line_length_px * MICRONS_PER_PIXEL / 1000

        # Count intersections: transitions from 0→255 along the line
        intersections = 0
        in_boundary = False
        for px in line_pixels:
            if px > 127 and not in_boundary:
                intersections += 1
                in_boundary = True
            elif px <= 127:
                in_boundary = False

        # Check for triple junctions (local neighborhood check)
        # For simplicity: weight all intersections as 1.0
        # A full triple junction check would add 0.5 per junction found
        weighted_intersections = intersections  # base count

        total_length_mm     += line_length_mm
        total_intersections += weighted_intersections

        results.append({
            "type"         : line_type,
            "position"     : pos,
            "intersections": intersections,
            "length_mm"    : line_length_mm,
        })

    # ── ASTM Calculation ─────────────────────────────────
    if total_intersections == 0:
        print("Warning: No intersections found!")
        return None

    # Mean intercept length in mm
    mean_intercept_mm = total_length_mm / total_intersections

    # ASTM Grain Size Number
    # G = (6.643856 * log10(l_bar)) - 3.288  where l_bar is in mm
    import math
    G = (6.643856 * math.log10(mean_intercept_mm)) - 3.288

    print(f"\n{'='*50}")
    print(f"ASTM E112 Analysis: {image_filename}")
    print(f"{'='*50}")
    print(f"  Total test line length : {total_length_mm:.4f} mm")
    print(f"  Total intersections    : {total_intersections}")
    print(f"  Mean intercept length  : {mean_intercept_mm:.6f} mm")
    print(f"  ASTM Grain Size Number : G = {G:.2f}")
    print(f"{'='*50}")

    return {
        "mean_intercept_mm": mean_intercept_mm,
        "G"                : G,
        "total_intersections": total_intersections,
        "test_lines"       : results,
    }


def visualize_analysis(original, skeleton, analysis_results, image_filename="image"):
    """Draw test lines and intersection points on the image."""

    if analysis_results is None:
        return

    h, w = skeleton.shape

    # Convert grayscale to BGR for colored overlay
    if len(original.shape) == 2:
        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        overlay = original.copy()

    # Draw test lines in blue
    for line in analysis_results["test_lines"]:
        if line["type"] == "H":
            cv2.line(overlay, (0, line["position"]),
                     (w, line["position"]), (255, 100, 0), 1)
        else:
            cv2.line(overlay, (line["position"], 0),
                     (line["position"], h), (255, 100, 0), 1)

    # Mark intersections in red
    for line in analysis_results["test_lines"]:
        if line["type"] == "H":
            y = line["position"]
            line_pixels = skeleton[y, :]
            in_boundary = False
            for x, px in enumerate(line_pixels):
                if px > 127 and not in_boundary:
                    cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
                    in_boundary = True
                elif px <= 127:
                    in_boundary = False
        else:
            x = line["position"]
            line_pixels = skeleton[:, x]
            in_boundary = False
            for y, px in enumerate(line_pixels):
                if px > 127 and not in_boundary:
                    cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
                    in_boundary = True
                elif px <= 127:
                    in_boundary = False

    # Add ASTM number on image
    G = analysis_results["G"]
    cv2.putText(overlay, f"G = {G:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Enhanced")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"ASTM Analysis — G = {G:.2f}")
    plt.axis("off")

    plt.suptitle(f"Stage 5 - ASTM E112 | {image_filename}")
    plt.tight_layout()
    plt.show()
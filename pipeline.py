# pipeline.py

import os
import cv2
from stages.preprocessing import preprocess
from stages.binarization import binarize
from stages.morphology import apply_morphology
from stages.segmentation import segment
from stages.analysis import analyze
from utils.visualize import show_stages, show_segmentation, save_image
from utils.metrics import compare_with_ground_truth
from config import IMAGES_DIR, OUTPUT_DIR

def run_pipeline(image_filename, show_steps=False):
    """
    Run the full pipeline on a single image.
    Returns all intermediate results + ASTM analysis.
    """

    image_path = os.path.join(IMAGES_DIR, image_filename)
    print(f"\n{'='*50}")
    print(f"Processing: {image_filename}")
    print(f"{'='*50}")

    # Stage 1 - Preprocessing
    print("Stage 1: Preprocessing...")
    original, gray, denoised, enhanced = preprocess(image_path)

    # Stage 2 - Binarization
    print("Stage 2: Binarization...")
    binary = binarize(enhanced)

    # Stage 3 - Morphology
    print("Stage 3: Morphology...")
    cleaned, skeleton = apply_morphology(binary)

    # Stage 4 - Segmentation
    print("Stage 4: Segmentation...")
    closed_skeleton, markers = segment(skeleton, enhanced)

    # Stage 5 - ASTM Analysis
    print("Stage 5: ASTM Analysis...")
    astm_results = analyze(skeleton, image_filename)

    # Save outputs
    name = os.path.splitext(image_filename)[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_image(skeleton, f"{name}_skeleton.png")
    save_image(binary,   f"{name}_binary.png")

    # Metrics vs ground truth
    metrics = compare_with_ground_truth(skeleton, image_filename)

    print("Done!")
    return enhanced, skeleton, markers, astm_results, metrics


if __name__ == "__main__":

    images = [
        "A_01_01.png", "A_01_02.png", "A_01_03.png",
        "A_02_01.png", "A_02_02.png",
        "A_03_01.png", "A_03_02.png", "A_03_03.png",
        "A_04_01.png", "A_04_02.png", "A_04_03.png", "A_04_04.png",
        "B_01_01.png", "B_01_02.png", "B_01_03.png",
    ]

    print(f"Running pipeline on {len(images)} images...\n")

    summary = []
    for img_file in images:
        enhanced, skeleton, markers, astm, metrics = run_pipeline(img_file)
        if astm and metrics:
            summary.append({
                "image"    : img_file,
                "G"        : astm["G"],
                "intercept": astm["mean_intercept_mm"],
                "precision": metrics[0],
                "recall"   : metrics[1],
                "f1"       : metrics[2],
            })

    # Final summary table
    print(f"\n{'='*75}")
    print(f"{'FINAL SUMMARY':^75}")
    print(f"{'='*75}")
    print(f"{'Image':<20} {'G':>6} {'Intercept(mm)':>15} {'P':>7} {'R':>7} {'F1':>7}")
    print(f"{'-'*75}")
    for r in summary:
        print(f"{r['image']:<20} {r['G']:>6.2f} {r['intercept']:>15.6f} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}")

    if summary:
        avg_G  = sum(r["G"]  for r in summary) / len(summary)
        avg_f1 = sum(r["f1"] for r in summary) / len(summary)
        print(f"{'-'*75}")
        print(f"{'AVERAGE':<20} {avg_G:>6.2f} {'':>15} {'':>7} {'':>7} {avg_f1:>7.3f}")
    print(f"{'='*75}")
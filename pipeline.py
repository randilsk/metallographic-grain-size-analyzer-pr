# pipeline.py

import os
import cv2
from stages.preprocessing import preprocess
from stages.binarization import binarize
from stages.morphology import apply_morphology
from stages.segmentation import segment
from utils.visualize import show_stages, show_segmentation, save_image
from config import IMAGES_DIR, OUTPUT_DIR

def run_pipeline(image_filename, show_steps=True):
    """
    Run the full pipeline on a single image.
    Returns all intermediate results + final markers.
    """

    image_path = os.path.join(IMAGES_DIR, image_filename)
    print(f"\n{'='*50}")
    print(f"Processing: {image_filename}")
    print(f"{'='*50}")

    # Stage 1 - Preprocessing
    print("Stage 1: Preprocessing...")
    original, gray, denoised, enhanced = preprocess(image_path)
    if show_steps:
        show_stages({
            "Original" : original,
            "Grayscale": gray,
            "Denoised" : denoised,
            "Enhanced" : enhanced,
        }, title=f"Stage 1 - Preprocessing | {image_filename}")

    # Stage 2 - Binarization
    print("Stage 2: Binarization...")
    binary = binarize(enhanced)
    if show_steps:
        show_stages({
            "Enhanced": enhanced,
            "Binary"  : binary,
        }, title=f"Stage 2 - Binarization | {image_filename}")

    # Stage 3 - Morphology
    print("Stage 3: Morphology...")
    cleaned, skeleton = apply_morphology(binary)
    if show_steps:
        show_stages({
            "Binary"  : binary,
            "Cleaned" : cleaned,
            "Skeleton": skeleton,
        }, title=f"Stage 3 - Morphology | {image_filename}")

    # Stage 4 - Segmentation
    print("Stage 4: Segmentation...")
    closed_skeleton, markers = segment(skeleton, enhanced)
    if show_steps:
        show_segmentation(enhanced, markers,
            title=f"Stage 4 - Segmentation | {image_filename}")

    # Save key outputs
    name = os.path.splitext(image_filename)[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_image(skeleton, f"{name}_skeleton.png")
    save_image(binary,   f"{name}_binary.png")

    print("Done!")
    return enhanced, skeleton, markers


if __name__ == "__main__":
    # Run on all images in the images/ folder
    images = sorted([
        f for f in os.listdir(IMAGES_DIR)
        if f.endswith(".png") or f.endswith(".jpg")
    ])

    print(f"Found {len(images)} images")

    for img_file in images:
        enhanced, skeleton, markers = run_pipeline(
            img_file,
            show_steps=False   # set True to see every stage for every image
        )
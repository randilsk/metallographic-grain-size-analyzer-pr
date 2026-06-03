# utils/visualize.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from config import OUTPUT_DIR

def show_stages(images_dict, title="Pipeline Stages"):
    """
    Pass a dict like:
      {"Original": img1, "Grayscale": img2, "Denoised": img3}
    and it displays them all side by side.
    """
    n = len(images_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, images_dict.items()):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def save_image(img, filename):
    """Save an output image to the output/ folder."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(path, img)
    print(f"Saved: {path}")

def show_segmentation(enhanced, markers, title="Stage 4 - Segmentation"):
    """Colorize each grain region with a random color."""
    import random
    h, w = enhanced.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(markers)
    for label in unique_labels:
        if label <= 0:  # skip background and boundary (-1)
            continue
        mask = markers == label
        color = [random.randint(50, 255) for _ in range(3)]
        colored[mask] = color

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(enhanced, cmap="gray")
    plt.title("Enhanced")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(colored)
    plt.title("Segmented Grains")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
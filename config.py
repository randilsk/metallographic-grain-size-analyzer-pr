import os

# ── Paths ────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR      = os.path.join(BASE_DIR, "images")
SEGMENTED_DIR   = os.path.join(BASE_DIR, "segmented")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")

# ── Stage 1: Preprocessing ───────────────────────────────
GAUSSIAN_KERNEL  = (7, 7)   # noise reduction kernel size
CLAHE_CLIP_LIMIT = 2.0      # contrast enhancement strength
CLAHE_TILE_SIZE  = (8, 8)   # local region size for CLAHE

# ── Stage 2: Binarization ────────────────────────────────
ADAPTIVE_BLOCK_SIZE = 25    # local window size (must be odd); larger = less interior noise
ADAPTIVE_C          = 5     # constant subtracted from mean; higher = stricter threshold
CANNY_LOW           = 50    # Canny lower hysteresis threshold
CANNY_HIGH          = 150   # Canny upper hysteresis threshold

# ── Stage 3: Morphology ──────────────────────────────────
MORPH_KERNEL_SIZE   = (3, 3)
MIN_AREA_THRESHOLD  = 80    # remove noise smaller than this (pixels); removes scratch artifacts
PRUNE_LENGTH        = 3     # remove skeleton branches shorter than this

# ── Stage 4: Segmentation ────────────────────────────────
GAP_CLOSE_DISTANCE  = 10    # max pixel gap to bridge in boundaries

# ── Stage 5: Analysis ────────────────────────────────────
MICRONS_PER_PIXEL = 100 / 226   # update from your scale bar
NUM_TEST_LINES      = 10    # intercept lines to draw
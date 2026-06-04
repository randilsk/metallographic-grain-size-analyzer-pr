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
ADAPTIVE_BLOCK_SIZE = 21    # local window size (must be odd)
ADAPTIVE_C          = 4     # constant subtracted from mean

# ── Stage 3: Morphology ──────────────────────────────────
MORPH_KERNEL_SIZE   = (3, 3)
MIN_AREA_THRESHOLD  = 30    # remove noise smaller than this (pixels)
PRUNE_LENGTH        = 5    # remove skeleton branches shorter than this

# ── Stage 4: Segmentation ────────────────────────────────
GAP_CLOSE_DISTANCE  = 10    # max pixel gap to bridge in boundaries

# ── Stage 5: Analysis ────────────────────────────────────
MICRONS_PER_PIXEL = 100 / 226   # update from your scale bar
NUM_TEST_LINES      = 10    # intercept lines to draw
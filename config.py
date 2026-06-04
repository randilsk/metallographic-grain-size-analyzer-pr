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

# NLM Denoising parameters
NLM_H            = 5        # filter strength (higher = more denoising)
NLM_TEMPLATE_WIN = 7        # template window size
NLM_SEARCH_WIN   = 21       # search window size

# ── Stage 2: Binarization ────────────────────────────────
ADAPTIVE_BLOCK_SIZE = 25    # local window size (must be odd); larger = less interior noise
ADAPTIVE_C          = 6     # constant subtracted from mean; higher = stricter threshold
CANNY_LOW           = 35    # Canny lower hysteresis threshold
CANNY_HIGH          = 110   # Canny upper hysteresis threshold
CANNY_DILATION_SIZE = 15    # dilation radius for Canny validation mask

# ── Stage 3: Morphology ──────────────────────────────────
MORPH_KERNEL_SIZE   = (3, 3)
CLOSE_KERNEL_SIZE   = (3, 3)  # closing kernel to bridge boundary gaps
MIN_AREA_THRESHOLD  = 120    # remove noise smaller than this (pixels); was 80
PRUNE_LENGTH        = 7      # remove skeleton branches shorter than this (was 3)
MIN_SKELETON_CC     = 40     # minimum connected component size in skeleton (pixels)

# ── Stage 4: Segmentation ────────────────────────────────
GAP_CLOSE_DISTANCE  = 10    # max pixel gap to bridge in boundaries

# ── Stage 5: Analysis ────────────────────────────────────
MICRONS_PER_PIXEL = 100 / 226   # update from your scale bar
NUM_TEST_LINES      = 10    # intercept lines to draw
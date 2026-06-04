# test_stage5.py

import os
from stages.preprocessing import preprocess
from stages.binarization import binarize
from stages.morphology import apply_morphology
from stages.segmentation import segment
from stages.analysis import analyze, visualize_analysis

image_path = os.path.join("images", "A_01_01.png")

original, gray, denoised, enhanced = preprocess(image_path)
binary = binarize(enhanced)
cleaned, skeleton = apply_morphology(binary)
closed_skeleton, markers = segment(skeleton, enhanced)
results = analyze(skeleton, "A_01_01.png")
visualize_analysis(enhanced, skeleton, results, "A_01_01.png")
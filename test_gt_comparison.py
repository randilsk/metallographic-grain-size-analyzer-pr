# test_gt_comparison.py

import os
from pipeline import run_pipeline
from utils.ground_truth_analysis import analyze_ground_truth

images = [
    "A_01_01.png", "A_01_02.png", "A_01_03.png",
    "A_02_01.png", "A_02_02.png",
    "A_03_01.png", "A_03_02.png", "A_03_03.png",
    "A_04_01.png", "A_04_02.png", "A_04_03.png", "A_04_04.png",
    "B_01_01.png", "B_01_02.png", "B_01_03.png",
]

print(f"\n{'='*70}")
print(f"{'G VALUE COMPARISON: Pipeline vs Ground Truth':^70}")
print(f"{'='*70}")
print(f"{'Image':<20} {'G Pipeline':>12} {'G GroundTruth':>14} {'Error':>8} {'Error%':>8}")
print(f"{'-'*70}")

errors = []
for img in images:
    # Run pipeline
    enhanced, skeleton, markers, astm, metrics = run_pipeline(
        img, show_steps=False
    )

    if astm is None:
        continue

    # Get ground truth G
    gt_G = analyze_ground_truth(img)

    if gt_G is None:
        print(f"{img:<20} {astm['G']:>12.2f} {'N/A':>14}")
        continue

    error    = abs(astm["G"] - gt_G)
    error_pct = (error / abs(gt_G)) * 100
    errors.append(error_pct)

    # Flag large errors
    flag = " ⚠" if error_pct > 15 else ""

    print(f"{img:<20} {astm['G']:>12.2f} {gt_G:>14.2f} "
          f"{error:>8.2f} {error_pct:>7.1f}%{flag}")

print(f"{'-'*70}")
if errors:
    print(f"{'AVERAGE ERROR':<20} {'':>12} {'':>14} "
          f"{sum(errors)/len(errors):>8.2f} "
          f"{sum(errors)/len(errors):>7.1f}%")
print(f"{'='*70}")
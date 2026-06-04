# test_pipeline.py

from pipeline import run_pipeline
from utils.metrics import compare_with_ground_truth

# Test on 5 images across different groups
test_images = [
    "A_01_01.png",
    "A_02_01.png",
    "A_03_01.png",
    "B_01_01.png",
    "B_01_02.png",
]

results = []
for img in test_images:
    enhanced, skeleton, markers,astm,metrics = run_pipeline(img, show_steps=False)
    metrics = compare_with_ground_truth(skeleton, img)
    if metrics:
        results.append((img, *metrics))

# Summary
print(f"\n{'='*50}")
print("SUMMARY")
print(f"{'='*50}")
for name, p, r, f1 in results:
    print(f"{name:20s} | P: {p:.3f} | R: {r:.3f} | F1: {f1:.3f}")
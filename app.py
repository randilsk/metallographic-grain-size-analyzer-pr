# app.py

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import random
from PIL import Image

from stages.preprocessing import preprocess
from stages.binarization import binarize
from stages.morphology import apply_morphology
from stages.segmentation import segment
from stages.analysis import analyze, visualize_analysis
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for streamlit

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Metallographic Grain Size Analyzer",
    page_icon="🔬",
    layout="wide"
)

# ── Header ───────────────────────────────────────────────
st.title("🔬 Metallographic Grain Size Analyzer")
st.markdown("Upload a microscopic image to automatically analyze grain size using **ASTM E112** intercept method.")
st.divider()

# ── Sidebar controls ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    show_stages = st.checkbox("Show intermediate stages", value=True)
    num_test_lines = st.slider("Number of test lines", 5, 20, 10)
    st.divider()
    st.markdown("**About**")
    st.markdown("Classical image processing pipeline for metallographic grain size measurement.")
    st.markdown("- Bilateral filter + CLAHE preprocessing")
    st.markdown("- Adaptive threshold + Canny edge fusion")
    st.markdown("- Morphological skeletonization")
    st.markdown("- Watershed segmentation")
    st.markdown("- ASTM E112 intercept analysis")

# ── File uploader ─────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload microscope image",
    type=["png", "jpg", "jpeg", "bmp", "tif"],
    help="Supported formats: PNG, JPG, BMP, TIF"
)

if uploaded is not None:

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.success(f"Loaded: {uploaded.name}")
    st.divider()

    # ── Run pipeline ─────────────────────────────────────
    with st.spinner("Running pipeline..."):

        # Stage 1
        original, gray, denoised, enhanced = preprocess(tmp_path)

        # Stage 2
        binary = binarize(enhanced)

        # Stage 3
        cleaned, skeleton = apply_morphology(binary)

        # Stage 4
        closed_skeleton, markers = segment(skeleton, enhanced)

        # Stage 5
        # Override NUM_TEST_LINES with slider value
        import config
        original_ntl = config.NUM_TEST_LINES
        config.NUM_TEST_LINES = num_test_lines
        astm = analyze(skeleton, uploaded.name)
        config.NUM_TEST_LINES = original_ntl

    # ── Results header ────────────────────────────────────
    st.subheader("📊 Results")

    if astm:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ASTM Grain Size (G)", f"{astm['G']:.2f}")
        col2.metric("Mean Intercept Length", f"{astm['mean_intercept_mm']*1000:.2f} µm")
        col3.metric("Total Intersections", astm['total_intersections'])
        col4.metric("Test Line Length", f"{astm['mean_intercept_mm'] * astm['total_intersections']:.3f} mm")

    st.divider()

    # ── Image display ─────────────────────────────────────
    if show_stages:
        st.subheader("🔄 Pipeline Stages")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Original**")
            st.image(
                cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        with col2:
            st.markdown("**Enhanced**")
            st.image(enhanced, use_container_width=True, clamp=True)

        with col3:
            st.markdown("**Binary**")
            st.image(binary, use_container_width=True, clamp=True)

        with col4:
            st.markdown("**Skeleton**")
            st.image(skeleton, use_container_width=True, clamp=True)

    st.divider()

    # ── Segmentation view ─────────────────────────────────
    st.subheader("🎨 Grain Segmentation")

    h, w = enhanced.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    unique_labels = np.unique(markers)
    for label in unique_labels:
        if label <= 0:
            continue
        mask = markers == label
        color = [random.randint(50, 255) for _ in range(3)]
        colored[mask] = color

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )
    with col2:
        st.markdown("**Segmented Grains**")
        st.image(colored, use_container_width=True)

    st.divider()

    # ── ASTM overlay ──────────────────────────────────────
    st.subheader("📏 ASTM E112 Analysis")

    if astm:
        # Build overlay image
        overlay = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Draw test lines in blue
        for line in astm["test_lines"]:
            if line["type"] == "H":
                cv2.line(overlay, (0, line["position"]),
                         (w, line["position"]), (255, 100, 0), 1)
            else:
                cv2.line(overlay, (line["position"], 0),
                         (line["position"], h), (255, 100, 0), 1)

        # Mark intersections in red
        for line in astm["test_lines"]:
            if line["type"] == "H":
                y = line["position"]
                lp = skeleton[y, :]
                in_b = False
                for x, px in enumerate(lp):
                    if px > 127 and not in_b:
                        cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
                        in_b = True
                    elif px <= 127:
                        in_b = False
            else:
                x = line["position"]
                lp = skeleton[:, x]
                in_b = False
                for y, px in enumerate(lp):
                    if px > 127 and not in_b:
                        cv2.circle(overlay, (x, y), 3, (0, 0, 255), -1)
                        in_b = True
                    elif px <= 127:
                        in_b = False

        # Add G number text
        cv2.putText(overlay, f"G = {astm['G']:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Skeleton**")
            st.image(skeleton, use_container_width=True, clamp=True)
        with col2:
            st.markdown("**ASTM Test Lines + Intersections**")
            st.image(
                cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        st.divider()

        # ── Download ──────────────────────────────────────
        st.subheader("💾 Download Results")

        # Save overlay to bytes for download
        _, buffer = cv2.imencode(".png", overlay)
        st.download_button(
            label="Download ASTM Overlay Image",
            data=buffer.tobytes(),
            file_name=f"{os.path.splitext(uploaded.name)[0]}_astm.png",
            mime="image/png"
        )

    # Cleanup temp file
    os.unlink(tmp_path)

else:
    # Landing state
    st.info("👆 Upload a metallographic microscope image to get started.")
    st.markdown("""
    **What this tool does:**
    - Detects grain boundaries using classical image processing
    - Segments individual grains using watershed algorithm
    - Calculates ASTM E112 grain size number using the intercept method
    - Visualizes test lines and intersection points

    **Expected input:** Optical microscope images of metallic samples (PNG, JPG, TIF)
    """)
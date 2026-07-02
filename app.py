"""
Property Near The Beach Predictor — Streamlit app.

Serves the trained scikit-learn pipeline (models/beach_predictor.joblib).
Two modes:
  1. Upload one or more property images.
  2. Paste a Rightmove listing URL — images are scraped and classified.

A property is flagged BEACH FRONT if *any* of its images is predicted as beach.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.scraper import fetch_listing_images

MODEL_FILE = Path(__file__).resolve().parent / "models" / "beach_predictor.joblib"

# ── Feature extraction ───────────────────────────────────────────────────────
# Must stay identical to notebooks/02_transform.ipynb: the model was trained on
# this exact 113-dim colour feature vector.
#
#     RGB means (3) + RGB stds (3) + HSV means (3)
#     + RGB histogram 32 bins x 3 channels (96) + 8 hue buckets (8) = 113
# IMAGE_SIZE is the resize target every image is squashed to before feature
# extraction. The value (224, 224) is NOT arbitrary: notebooks/02_transform.ipynb
# scans the native width/height of every image in the dataset and takes the
# MEDIAN of each dimension, rounded down to the nearest multiple of 16
# (median width = 224, median height = 224 → min(224, 224) = 224). It is
# hardcoded here so the app resizes inputs exactly as the training pipeline did.
IMAGE_SIZE = (224, 224)   # derived median dimension — see 02_transform.ipynb
HIST_BINS = 32            # bins per channel in the colour histogram

# Column names (order MUST match the assembled feature vector).
_rgb_mean_cols   = ["rgb_mean_r", "rgb_mean_g", "rgb_mean_b"]
_rgb_std_cols    = ["rgb_std_r",  "rgb_std_g",  "rgb_std_b"]
_hsv_mean_cols   = ["hsv_mean_h", "hsv_mean_s", "hsv_mean_v"]
_hist_cols       = [f"hist_{ch}_{i}" for ch in ("r", "g", "b") for i in range(HIST_BINS)]
_hue_bucket_cols = ["hue_red", "hue_orange", "hue_yellow", "hue_green",
                    "hue_cyan", "hue_blue",   "hue_indigo", "hue_violet"]
FEATURE_COLS = _rgb_mean_cols + _rgb_std_cols + _hsv_mean_cols + _hist_cols + _hue_bucket_cols


def _rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    """Convert a float32 (H, W, 3) RGB array in [0, 1] to HSV in [0, 1]."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc    = np.maximum(np.maximum(r, g), b)
    minc    = np.minimum(np.minimum(r, g), b)
    delta   = maxc - minc
    v = maxc
    s = np.where(maxc > 0, delta / maxc, 0.0)
    safe_d = np.where(delta > 0, delta, 1.0)          # avoid /0
    h_r = ((g - b) / safe_d) % 6
    h_g = (b - r) / safe_d + 2
    h_b = (r - g) / safe_d + 4
    h   = np.where(delta > 0,
            np.where(maxc == r, h_r,
            np.where(maxc == g, h_g, h_b)),
            0.0) / 6.0 % 1.0
    return np.stack([h, s, v], axis=-1)


def extract_features(img: Image.Image) -> list[float]:
    """Return the 113-dim feature list for a single PIL image."""
    img = img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0    # (224, 224, 3) in [0, 1]

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    rgb_means = [float(r.mean()), float(g.mean()), float(b.mean())]
    rgb_stds  = [float(r.std()),  float(g.std()),  float(b.std())]

    hsv       = _rgb_to_hsv(arr)
    hsv_means = hsv.mean(axis=(0, 1)).tolist()

    n_px  = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    hists = np.concatenate([
        np.histogram(ch.ravel(), bins=HIST_BINS, range=(0.0, 1.0))[0]
        for ch in (r, g, b)
    ]).astype(np.float32) / n_px

    sat_mask = hsv[..., 1] > 0.15
    if sat_mask.sum() > 0:
        hue_buckets = (
            np.histogram(hsv[..., 0][sat_mask], bins=8, range=(0.0, 1.0))[0]
            .astype(np.float32) / sat_mask.sum()
        )
    else:
        hue_buckets = np.zeros(8, dtype=np.float32)

    return rgb_means + rgb_stds + hsv_means + hists.tolist() + hue_buckets.tolist()


st.set_page_config(page_title="Property Near The Beach Predictor", page_icon="🏖️", layout="wide")


@st.cache_resource(show_spinner="Loading model …")
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def predict_images(model, images: list[Image.Image]) -> pd.DataFrame:
    """Extract features for each image and return per-image predictions."""
    rows = []
    for img in images:
        feats = extract_features(img)
        X = pd.DataFrame([feats], columns=FEATURE_COLS)
        pred = int(model.predict(X)[0])
        # Probability of the beach class (1), if the estimator supports it
        try:
            proba = float(model.predict_proba(X)[0][list(model.classes_).index(1)])
        except (AttributeError, ValueError):
            proba = float("nan")
        rows.append({"prediction": pred, "beach_probability": proba})
    return pd.DataFrame(rows)


def render_results(images: list[Image.Image], results: pd.DataFrame) -> None:
    is_beach = bool((results["prediction"] == 1).any())
    if is_beach:
        st.success("🏖️  BEACH FRONT — this property appears to have beach views")
    else:
        st.info("🏠  NOT BEACH FRONT — no coastal images detected")

    cols = st.columns(4)
    for i, (img, (_, row)) in enumerate(zip(images, results.iterrows())):
        with cols[i % 4]:
            label = "🏖️ Beach" if row["prediction"] == 1 else "🏠 Not beach"
            caption = label
            if not np.isnan(row["beach_probability"]):
                caption += f"  ·  {row['beach_probability']:.0%}"
            st.image(img, caption=caption, use_container_width=True)


def main() -> None:
    st.title("🏖️ Property Near The Beach Predictor")
    st.caption(
        "Predicts whether a property is beachfront from its photos, using a "
        "colour-feature classifier trained with scikit-learn."
    )

    model = load_model()
    if model is None:
        st.error(
            f"Model file not found at `{MODEL_FILE}`.\n\n"
            "Run the notebooks (01–05) to train and save "
            "`models/beach_predictor.joblib`, and make sure it is included in "
            "the deployment."
        )
        st.stop()

    mode = st.radio("Input mode", ["Upload images", "Rightmove URL"], horizontal=True)

    if mode == "Upload images":
        uploads = st.file_uploader(
            "Upload property image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploads:
            images = [Image.open(f).convert("RGB") for f in uploads]
            with st.spinner("Classifying …"):
                results = predict_images(model, images)
            render_results(images, results)

    else:  # Rightmove URL
        url = st.text_input("Rightmove listing URL", placeholder="https://www.rightmove.co.uk/properties/...")
        if st.button("Analyse listing", type="primary") and url:
            with st.spinner("Fetching and classifying listing images …"):
                try:
                    images = fetch_listing_images(url)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not fetch the listing: {exc}")
                    st.stop()
                if not images:
                    st.warning("No images could be extracted from that URL.")
                    st.stop()
                results = predict_images(model, images)
            render_results(images, results)


if __name__ == "__main__":
    main()

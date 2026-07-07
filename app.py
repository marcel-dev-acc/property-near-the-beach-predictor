"""
Property Near The Beach Predictor — Streamlit app.

Serves the trained scikit-learn pipeline (models/beach_predictor.joblib).
Two modes:
  1. Upload one or more property images.
  2. Paste a Rightmove listing URL — images are scraped and classified.

A property is flagged BEACH FRONT if any listing image is predicted as beach.

Run locally:   streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.image_features import ImageFeatureConfig, extract_feature_frame
from src.model_bundle import predict_with_bundle, unpack_model_bundle
from src.scraper import fetch_listing_images

MODEL_FILE = Path(__file__).resolve().parent / "models" / "beach_predictor.joblib"


st.set_page_config(page_title="Property Near The Beach Predictor", page_icon="🏖️", layout="wide")


@st.cache_resource(show_spinner="Loading model …")
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def predict_images(bundle_or_model, images: list[Image.Image]) -> pd.DataFrame:
    unpacked = unpack_model_bundle(bundle_or_model)
    config = ImageFeatureConfig(**unpacked["image_config"])
    X = extract_feature_frame(
        images,
        config=config,
        feature_columns=unpacked["feature_columns"],
    )
    preds, probs = predict_with_bundle(bundle_or_model, X)
    return pd.DataFrame({"prediction": preds, "beach_probability": probs})


def render_results(images: list[Image.Image], results: pd.DataFrame) -> None:
    is_beach = bool((results["prediction"] == 1).any())
    if is_beach:
        st.success("🏖️  BEACH FRONT — this property appears to have beach views")
    else:
        st.info("🏠  NOT BEACH FRONT — no strong coastal evidence detected")

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

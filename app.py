"""
Property Near The Beach Predictor dashboard.

Run locally with:
    streamlit run app.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.image_features import ImageFeatureConfig, extract_feature_frame
from src.model_bundle import build_preprocessor, predict_with_bundle, unpack_model_bundle
from src.scraper import fetch_listing_images

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_FILE = ROOT_DIR / "models" / "beach_predictor.joblib"
TRAIN_FILE = DATA_DIR / "processed" / "train.parquet"
TEST_FILE = DATA_DIR / "processed" / "test.parquet"

TARGET_COLUMN = "label"
IMAGE_PATH_COLUMN = "filepath"
GROUP_COLUMN = "class_name"
MODEL_SELECT_K = 60
EDA_FEATURE_COUNT = 12
CORRELATION_FEATURE_COUNT = 6
MONTAGE_SAMPLES_PER_CLASS = 4

PAGE_OPTIONS = [
    "Introduction",
    "Business Case & Hypothesis",
    "Analysis",
    "Prediction",
    "Conclusions",
]

CLASS_LABELS = {1: "Beach", 0: "Not beach"}
CLASS_ICONS = {1: "Beach", 0: "Not beach"}


st.set_page_config(
    page_title="Property Near The Beach Predictor",
    page_icon="🏖️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_dataframe(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def get_model_details() -> dict[str, object] | None:
    if not MODEL_FILE.exists():
        return None

    bundle = joblib.load(MODEL_FILE)
    unpacked = unpack_model_bundle(bundle)
    selected_raw = list(unpacked.get("selected_feature_names", []))
    selected_clean = [strip_feature_prefix(name) for name in selected_raw]

    importances = pd.Series(dtype=float)
    family_counts = pd.Series(dtype=int)
    if selected_clean:
        importances = pd.Series(
            unpacked["model"].named_steps["clf"].feature_importances_,
            index=selected_clean,
            name="importance",
        ).sort_values(ascending=False)
        family_counts = pd.Series(
            [name.split("__", 1)[0] if "__" in name else "other" for name in selected_raw],
            name="count",
        ).value_counts()

    return {
        "feature_columns": list(unpacked.get("feature_columns", [])),
        "selected_raw": selected_raw,
        "selected_clean": selected_clean,
        "importances": importances,
        "family_counts": family_counts,
    }


@st.cache_data(show_spinner=False)
def get_evaluation_metrics() -> dict[str, object] | None:
    if not MODEL_FILE.exists() or not TEST_FILE.exists():
        return None

    bundle = joblib.load(MODEL_FILE)
    unpacked = unpack_model_bundle(bundle)
    test_df = pd.read_parquet(TEST_FILE)
    X_test = test_df[unpacked["feature_columns"]]
    y_test = test_df[TARGET_COLUMN]
    preds, probs = predict_with_bundle(bundle, X_test)
    matrix = confusion_matrix(y_test, preds)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "confusion_matrix": matrix,
        "support": int(len(test_df)),
        "true_negatives": int(matrix[0, 0]),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "true_positives": int(matrix[1, 1]),
    }


@st.cache_data(show_spinner="Preparing analysis summaries ...")
def compute_analysis_artifacts(train_df: pd.DataFrame) -> dict[str, object]:
    excluded_cols = {TARGET_COLUMN, IMAGE_PATH_COLUMN, GROUP_COLUMN}
    candidate_feature_cols = [
        col
        for col in train_df.columns
        if col not in excluded_cols and pd.api.types.is_numeric_dtype(train_df[col])
    ]

    X = train_df[candidate_feature_cols]
    y = train_df[TARGET_COLUMN]

    prep = build_preprocessor(candidate_feature_cols)
    X_prep = prep.fit_transform(X)
    prep_feature_names = pd.Index(prep.get_feature_names_out())
    clean_feature_names = prep_feature_names.str.split("__", n=1).str[-1]

    variance_filter = VarianceThreshold()
    X_var = variance_filter.fit_transform(X_prep)
    var_feature_names = clean_feature_names[variance_filter.get_support()]

    select_k = min(MODEL_SELECT_K, len(var_feature_names))
    selector = SelectKBest(mutual_info_classif, k=select_k)
    selector.fit(X_var, y)

    selected_mask = selector.get_support()
    selected_feature_scores = (
        pd.Series(
            selector.scores_[selected_mask],
            index=var_feature_names[selected_mask],
            name="mutual_information",
        )
        .sort_values(ascending=False)
        .dropna()
    )

    eda_feature_cols = selected_feature_scores.head(
        min(EDA_FEATURE_COUNT, len(selected_feature_scores))
    ).index.tolist()
    corr_feature_cols = eda_feature_cols[: min(CORRELATION_FEATURE_COUNT, len(eda_feature_cols))]
    corr_matrix = train_df[corr_feature_cols].corr() if len(corr_feature_cols) >= 2 else pd.DataFrame()

    pair_strength = pd.DataFrame(
        [
            {
                "x_feature": x_col,
                "y_feature": y_col,
                "correlation": corr_matrix.loc[x_col, y_col],
                "abs_correlation": abs(corr_matrix.loc[x_col, y_col]),
            }
            for x_col, y_col in combinations(corr_feature_cols, 2)
        ]
    )
    if not pair_strength.empty:
        pair_strength = pair_strength.sort_values("abs_correlation", ascending=False)

    distribution_rows = []
    for col in eda_feature_cols:
        series = train_df[col].dropna()
        skew = float(series.skew())
        excess_kurtosis = float(series.kurt())
        abs_skew = abs(skew)

        if abs_skew < 0.5 and abs(excess_kurtosis) < 1.0:
            distribution_class = "approximately_normal"
        elif abs_skew < 1.0:
            distribution_class = "slightly_skewed"
        else:
            distribution_class = "highly_skewed"

        if skew > 0.5:
            direction = "right_skewed"
        elif skew < -0.5:
            direction = "left_skewed"
        else:
            direction = "symmetric"

        distribution_rows.append(
            {
                "feature": col,
                "mutual_information": float(selected_feature_scores.loc[col]),
                "skewness": skew,
                "excess_kurtosis": excess_kurtosis,
                "direction": direction,
                "class": distribution_class,
            }
        )

    distribution_summary = pd.DataFrame(distribution_rows).sort_values(
        ["mutual_information", "feature"],
        ascending=[False, True],
    )

    return {
        "candidate_feature_count": len(candidate_feature_cols),
        "selected_feature_scores": selected_feature_scores,
        "eda_feature_cols": eda_feature_cols,
        "corr_feature_cols": corr_feature_cols,
        "corr_matrix": corr_matrix,
        "pair_strength": pair_strength,
        "distribution_summary": distribution_summary,
    }


@st.cache_data(show_spinner=False)
def get_montage_samples(train_df: pd.DataFrame) -> dict[str, list[str]]:
    group_col = GROUP_COLUMN if GROUP_COLUMN in train_df.columns else TARGET_COLUMN
    samples: dict[str, list[str]] = {}

    for raw_group_name in sorted(train_df[group_col].dropna().unique()):
        group_rows = train_df[train_df[group_col] == raw_group_name]
        sampled = group_rows.sample(
            n=min(MONTAGE_SAMPLES_PER_CLASS, len(group_rows)),
            random_state=42,
        )
        display_name = prettify_group_name(raw_group_name)
        samples[display_name] = sampled[IMAGE_PATH_COLUMN].tolist()

    return samples


def strip_feature_prefix(name: str) -> str:
    return name.split("__", 1)[1] if "__" in name else name


def prettify_group_name(group_name: object) -> str:
    if isinstance(group_name, str):
        return group_name.replace("_", " ").title()
    return CLASS_LABELS.get(int(group_name), str(group_name))


def resolve_image_path(filepath: str) -> Path | None:
    candidate = Path(filepath)
    candidates = [candidate, ROOT_DIR / candidate]

    if candidate.parts and candidate.parts[0] == "data":
        candidates.append(DATA_DIR / Path(*candidate.parts[1:]))

    for image_path in candidates:
        if image_path.exists():
            return image_path

    return None


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


def build_histogram_figure(train_df: pd.DataFrame, feature_cols: list[str]):
    n_cols = 3
    n_rows = max(1, (len(feature_cols) + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, col in zip(axes, feature_cols):
        train_df[col].hist(bins=30, ax=ax, color="#4C78A8", edgecolor="white")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    for ax in axes[len(feature_cols) :]:
        ax.axis("off")

    fig.suptitle("Top model-selected feature distributions", fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def build_correlation_heatmap(corr_matrix: pd.DataFrame):
    fig, ax = plt.subplots(
        figsize=(1.8 * len(corr_matrix.columns) + 2, 1.5 * len(corr_matrix.columns) + 2)
    )
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.index)

    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            ax.text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    ax.set_title("Correlation heatmap for the top model-selected features")
    plt.tight_layout()
    return fig


def build_pair_scatter_figure(train_df: pd.DataFrame, x_col: str, y_col: str):
    labels = sorted(train_df[TARGET_COLUMN].dropna().unique())
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(labels), 1)))
    label_to_color = {label: colors[idx] for idx, label in enumerate(labels)}

    plot_df = train_df[[x_col, y_col, TARGET_COLUMN]].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))

    for label in labels:
        subset = plot_df[plot_df[TARGET_COLUMN] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_col],
            subset[y_col],
            alpha=0.5,
            s=16,
            color=label_to_color[label],
            label=f"{TARGET_COLUMN}={label}",
        )

    if len(plot_df) >= 2:
        slope, intercept = np.polyfit(plot_df[x_col], plot_df[y_col], 1)
        x_line = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="black", linewidth=2, linestyle="--", label="Trend line")

    corr_value = plot_df[x_col].corr(plot_df[y_col])
    ax.set_title(f"{x_col} vs {y_col} (corr={corr_value:.2f})")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return fig


def build_feature_importance_figure(importances: pd.Series):
    top_importances = importances.head(12).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_importances.index, top_importances.values, color="#2B6CB0")
    ax.set_xlabel("Random forest importance")
    ax.set_title("Top chosen features in the final model")
    plt.tight_layout()
    return fig


def build_confusion_matrix_figure(matrix: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted not beach", "Predicted beach"])
    ax.set_yticklabels(["Actual not beach", "Actual beach"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=12)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Test-set confusion matrix")
    plt.tight_layout()
    return fig


def render_sidebar(model_loaded: bool) -> str:
    st.sidebar.title("Dashboard")
    st.sidebar.caption("Property Near The Beach Predictor")
    st.sidebar.success("Model loaded" if model_loaded else "Model not found")
    return st.sidebar.radio("Go to", PAGE_OPTIONS, index=0)


def render_introduction(
    train_df: pd.DataFrame | None,
    model_details: dict[str, object] | None,
    evaluation: dict[str, object] | None,
) -> None:
    st.title("Property Near The Beach Predictor")
    st.caption(
        "A Streamlit dashboard for exploring the dataset, reviewing the modelling "
        "approach, and predicting whether a property image suggests beach frontage."
    )

    left, right = st.columns([1.4, 1])
    with left:
        st.markdown(
            """
            This project uses colour, hue, and histogram-based image features to distinguish
            beach-facing scenes from non-beach property images. The dashboard brings the
            whole story into one place:

            - an introduction to the problem and why it matters
            - the business case and hypothesis behind the model
            - analysis visuals drawn from the training data
            - the selected features used by the classifier
            - a live prediction page for new uploads
            - a conclusion page summarising model performance
            """
        )
    with right:
        metric_cols = st.columns(2)
        if train_df is not None:
            metric_cols[0].metric("Training images", f"{len(train_df):,}")
            beach_count = int((train_df[GROUP_COLUMN] == "beach").sum()) if GROUP_COLUMN in train_df.columns else int((train_df[TARGET_COLUMN] == 1).sum())
            metric_cols[1].metric("Beach examples", f"{beach_count:,}")
        if model_details is not None:
            metric_cols[0].metric("Candidate features", len(model_details["feature_columns"]))
            metric_cols[1].metric("Chosen features", len(model_details["selected_clean"]))
        if evaluation is not None:
            metric_cols[0].metric("Accuracy", f"{evaluation['accuracy']:.1%}")
            metric_cols[1].metric("ROC AUC", f"{evaluation['roc_auc']:.3f}")

    st.subheader("Dashboard workflow")
    st.markdown(
        """
        1. Start with the business case to understand the problem the model is solving.
        2. Review the analysis page to see example images, histogram patterns, and correlations.
        3. Inspect the chosen features that survived model selection.
        4. Upload a new image on the prediction page to test the classifier.
        5. Finish on conclusions to review the held-out test performance and limitations.
        """
    )


def render_business_case() -> None:
    st.title("Business Case & Hypothesis")
    st.markdown(
        """
        Coastal properties can command a price premium, but browsing listing photos manually
        is slow and inconsistent. A lightweight image classifier can help agents, investors,
        and analysts quickly triage listings by identifying photos that show strong beach or
        seafront signals.
        """
    )

    st.subheader("Business case")
    st.markdown(
        """
        - Reduce the time spent manually scanning thousands of property photos.
        - Surface listings with likely beach views earlier in the decision process.
        - Provide a simple, explainable screening tool that can sit in front of more
          complex pricing or valuation models.
        - Demonstrate that even compact colour-based features can produce useful signals
          before moving to heavier computer-vision pipelines.
        """
    )

    st.subheader("Core hypothesis")
    st.markdown(
        """
        **Hypothesis:** images from properties near the beach will contain a distinguishable
        colour signature, especially in blue and cyan ranges, compared with non-beach
        property imagery such as interiors, roads, roofs, and street scenes.
        """
    )

    st.subheader("Why this feature approach")
    st.markdown(
        """
        - Mean RGB and HSV values summarise the overall colour balance of each image.
        - Channel histograms capture how colour intensity is distributed across the frame.
        - Hue buckets help separate beach-like scenes with sky and sea tones from more
          neutral or built-up scenes.
        - The final classifier then selects the most informative subset rather than using
          every available feature.
        """
    )


def render_analysis(
    train_df: pd.DataFrame | None,
    analysis: dict[str, object] | None,
    model_details: dict[str, object] | None,
) -> None:
    st.title("Analysis Description")

    if train_df is None or analysis is None:
        st.warning("Training data is not available, so the analysis page cannot be rendered.")
        return

    tabs = st.tabs(["Montage", "Histograms", "Correlations", "Chosen Features"])

    with tabs[0]:
        st.markdown(
            "A quick visual sample from the training split shows the broad difference between beach and non-beach imagery."
        )
        samples = get_montage_samples(train_df)
        for group_name, filepaths in samples.items():
            st.markdown(f"**{group_name}**")
            cols = st.columns(len(filepaths))
            for col, filepath in zip(cols, filepaths):
                image_path = resolve_image_path(filepath)
                with col:
                    if image_path is None:
                        st.warning("Image not found")
                    else:
                        st.image(str(image_path), use_container_width=True)

    with tabs[1]:
        st.markdown(
            "These histograms focus on the strongest features shortlisted by the same selection logic used in the model pipeline."
        )
        score_table = (
            analysis["selected_feature_scores"]
            .head(EDA_FEATURE_COUNT)
            .rename_axis("feature")
            .reset_index()
        )
        st.dataframe(score_table, use_container_width=True, hide_index=True)
        hist_fig = build_histogram_figure(train_df, analysis["eda_feature_cols"])
        st.pyplot(hist_fig, clear_figure=True)
        st.dataframe(analysis["distribution_summary"], use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown(
            "Correlation review is limited to a smaller set of high-value features so the dashboard stays readable."
        )
        if analysis["corr_matrix"].empty:
            st.info("Not enough shortlisted features were available to compute correlations.")
        else:
            heatmap_fig = build_correlation_heatmap(analysis["corr_matrix"])
            st.pyplot(heatmap_fig, clear_figure=True)

            strongest_pairs = analysis["pair_strength"].head(3)
            st.dataframe(
                strongest_pairs[["x_feature", "y_feature", "correlation"]],
                use_container_width=True,
                hide_index=True,
            )

            for _, row in strongest_pairs.iterrows():
                scatter_fig = build_pair_scatter_figure(train_df, row["x_feature"], row["y_feature"])
                st.pyplot(scatter_fig, clear_figure=True)

    with tabs[3]:
        st.markdown(
            "The final trained model keeps 60 selected features after preprocessing and filtering. The chart below shows which of those selected features matter most to the random forest."
        )

        if model_details is None:
            st.info("Model details are unavailable.")
        else:
            family_cols = st.columns(4)
            for idx, (family, count) in enumerate(model_details["family_counts"].items()):
                if idx >= 4:
                    break
                family_cols[idx].metric(f"{family} features", int(count))

            importance_fig = build_feature_importance_figure(model_details["importances"])
            st.pyplot(importance_fig, clear_figure=True)

            top_features = (
                model_details["importances"]
                .head(20)
                .rename_axis("feature")
                .reset_index(name="importance")
            )
            st.dataframe(top_features, use_container_width=True, hide_index=True)


def render_prediction_page(model) -> None:
    st.title("Prediction")
    st.markdown(
        "Upload one or more property images to score them with the trained classifier, or analyse a Rightmove URL using the scraper already started in the app."
    )

    upload_tab, listing_tab = st.tabs(["Upload images", "Rightmove URL"])

    with upload_tab:
        uploads = st.file_uploader(
            "Upload property image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )
        if uploads:
            images = [Image.open(file).convert("RGB") for file in uploads]
            with st.spinner("Classifying uploaded image(s) ..."):
                results = predict_images(model, images)
            render_prediction_results(images, results)

    with listing_tab:
        url = st.text_input(
            "Rightmove listing URL",
            placeholder="https://www.rightmove.co.uk/properties/...",
        )
        if st.button("Analyse listing", type="primary") and url:
            with st.spinner("Fetching and classifying listing images ..."):
                try:
                    images = fetch_listing_images(url)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not fetch the listing: {exc}")
                    return
                if not images:
                    st.warning("No images could be extracted from that URL.")
                    return
                results = predict_images(model, images)
            render_prediction_results(images, results)


def render_prediction_results(images: list[Image.Image], results: pd.DataFrame) -> None:
    is_beach = bool((results["prediction"] == 1).any())
    mean_probability = float(results["beach_probability"].mean())

    summary_cols = st.columns(3)
    summary_cols[0].metric("Images scored", len(results))
    summary_cols[1].metric("Average beach probability", f"{mean_probability:.1%}")
    summary_cols[2].metric(
        "Overall result",
        "Beach signal found" if is_beach else "No strong beach signal",
    )

    if is_beach:
        st.success("At least one uploaded image shows a strong beach-style visual signature.")
    else:
        st.info("The uploaded image set looks more like the non-beach examples seen during training.")

    result_cards = st.columns(min(3, len(images)))
    for idx, (image, (_, row)) in enumerate(zip(images, results.iterrows())):
        with result_cards[idx % len(result_cards)]:
            label = CLASS_ICONS[int(row["prediction"])]
            st.image(image, use_container_width=True)
            st.metric("Prediction", label)
            st.metric("Beach probability", f"{float(row['beach_probability']):.1%}")


def render_conclusions(
    evaluation: dict[str, object] | None,
    model_details: dict[str, object] | None,
) -> None:
    st.title("Conclusions")

    if evaluation is None:
        st.warning("Test metrics are unavailable because either the model or test split is missing.")
        return

    metric_cols = st.columns(5)
    metric_cols[0].metric("Accuracy", f"{evaluation['accuracy']:.1%}")
    metric_cols[1].metric("Precision", f"{evaluation['precision']:.1%}")
    metric_cols[2].metric("Recall", f"{evaluation['recall']:.1%}")
    metric_cols[3].metric("F1 score", f"{evaluation['f1']:.3f}")
    metric_cols[4].metric("ROC AUC", f"{evaluation['roc_auc']:.3f}")

    left, right = st.columns([1.1, 1])
    with left:
        matrix_fig = build_confusion_matrix_figure(evaluation["confusion_matrix"])
        st.pyplot(matrix_fig, clear_figure=True)
    with right:
        st.markdown(
            f"""
            On the held-out test set of **{evaluation['support']:,}** images, the model performs strongly overall.

            - It is especially precise when it predicts that an image is beach-related.
            - Recall is lower than precision, which means the model still misses some true beach images.
            - The strongest chosen features are dominated by blue-channel histogram bins and hue buckets,
              which matches the original hypothesis about sea and sky colour signals.
            """
        )

        if model_details is not None and not model_details["importances"].empty:
            top_three = ", ".join(model_details["importances"].head(3).index.tolist())
            st.markdown(f"**Most influential features:** {top_three}")

    st.subheader("Interpretation")
    st.markdown(
        """
        The model demonstrates that a simple, explainable colour-feature approach can already
        do a good job of separating beach and non-beach property imagery. It is best seen as a
        first-pass screening tool rather than a final decision-maker.
        """
    )

    st.subheader("Limitations and next steps")
    st.markdown(
        """
        - Some non-beach scenes still contain strong sky or water-like colour distributions.
        - Recall could likely improve with richer spatial features or a convolutional image model.
        - A future version could combine image predictions with listing text, geolocation, or price metadata.
        """
    )


def main() -> None:
    model = load_model()
    train_df = load_dataframe(str(TRAIN_FILE))
    model_details = get_model_details()
    evaluation = get_evaluation_metrics()
    analysis = compute_analysis_artifacts(train_df) if train_df is not None else None

    page = render_sidebar(model_loaded=model is not None)

    if page == "Introduction":
        render_introduction(train_df, model_details, evaluation)
        return

    if page == "Business Case & Hypothesis":
        render_business_case()
        return

    if page == "Analysis":
        render_analysis(train_df, analysis, model_details)
        return

    if page == "Prediction":
        if model is None:
            st.error(
                f"Model file not found at `{MODEL_FILE}`. Run the training notebooks first so predictions can be served."
            )
            return
        render_prediction_page(model)
        return

    if page == "Conclusions":
        render_conclusions(evaluation, model_details)


if __name__ == "__main__":
    main()

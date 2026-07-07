from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.image_features import DEFAULT_IMAGE_CONFIG, FEATURE_COLUMNS


def split_feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    return {
        "moments": [
            column
            for column in feature_columns
            if column.startswith(("rgb_mean_", "rgb_std_", "hsv_mean_"))
        ],
        "hist": [column for column in feature_columns if column.startswith("hist_")],
        "hue": [column for column in feature_columns if column.startswith("hue_")],
    }


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    groups = split_feature_groups(feature_columns)
    return ColumnTransformer(
        transformers=[
            ("moments", StandardScaler(), groups["moments"]),
            ("hist", StandardScaler(), groups["hist"]),
            ("hue", StandardScaler(), groups["hue"]),
        ],
        remainder="drop",
    )


def build_training_pipeline(
    feature_columns: list[str],
    *,
    select_k: int = 60,
    n_estimators: int = 300,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("prep", build_preprocessor(feature_columns)),
            ("var", VarianceThreshold()),
            ("select", SelectKBest(mutual_info_classif, k=select_k)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def get_selected_feature_names(pipeline: Pipeline) -> list[str]:
    prep = pipeline.named_steps["prep"]
    variance = pipeline.named_steps["var"]
    selector = pipeline.named_steps["select"]

    feature_names = np.asarray(prep.get_feature_names_out(), dtype=object)
    feature_names = feature_names[variance.get_support()]
    feature_names = feature_names[selector.get_support()]
    return feature_names.tolist()


def unpack_model_bundle(bundle_or_model: Any) -> dict[str, Any]:
    if isinstance(bundle_or_model, dict) and "model" in bundle_or_model:
        return {
            "model": bundle_or_model["model"],
            "feature_columns": list(
                bundle_or_model.get(
                    "feature_columns",
                    getattr(bundle_or_model["model"], "feature_names_in_", FEATURE_COLUMNS),
                )
            ),
            "image_config": bundle_or_model.get(
                "image_config",
                DEFAULT_IMAGE_CONFIG.to_dict(),
            ),
            "selected_feature_names": bundle_or_model.get("selected_feature_names", []),
        }

    return {
        "model": bundle_or_model,
        "feature_columns": list(getattr(bundle_or_model, "feature_names_in_", FEATURE_COLUMNS)),
        "image_config": DEFAULT_IMAGE_CONFIG.to_dict(),
        "selected_feature_names": [],
    }


def predict_with_bundle(bundle_or_model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    unpacked = unpack_model_bundle(bundle_or_model)
    model = unpacked["model"]
    probs = model.predict_proba(X)[:, list(model.classes_).index(1)]
    preds = model.predict(X)
    return preds, probs

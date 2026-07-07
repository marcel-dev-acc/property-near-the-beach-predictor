from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ALLOWED_NOT_BEACH_CATEGORIES = {
    "apartment",
    "bath",
    "bed",
    "din",
    "garage",
    "house",
    "kitchen",
    "living",
    "roof",
}


def infer_not_beach_category(filepath: str | Path) -> str:
    stem = Path(filepath).stem.lower()
    match = re.match(r"[a-z]+", stem)
    return match.group(0) if match else "unknown"


def filter_property_relevant_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    filtered = manifest.copy()
    filtered["not_beach_category"] = filtered["filepath"].map(infer_not_beach_category)
    keep_mask = (filtered["label"] == 1) | (
        filtered["not_beach_category"].isin(ALLOWED_NOT_BEACH_CATEGORIES)
    )
    filtered = filtered.loc[keep_mask].drop(columns=["not_beach_category"])
    return filtered.reset_index(drop=True)

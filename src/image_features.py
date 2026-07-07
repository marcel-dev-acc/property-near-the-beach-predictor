from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class ImageFeatureConfig:
    image_size: tuple[int, int] = (224, 224)
    hist_bins: int = 32
    saturation_threshold: float = 0.15

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_IMAGE_CONFIG = ImageFeatureConfig()

def build_feature_columns(config: ImageFeatureConfig = DEFAULT_IMAGE_CONFIG) -> list[str]:
    rgb_mean_cols = ["rgb_mean_r", "rgb_mean_g", "rgb_mean_b"]
    rgb_std_cols = ["rgb_std_r", "rgb_std_g", "rgb_std_b"]
    hsv_mean_cols = ["hsv_mean_h", "hsv_mean_s", "hsv_mean_v"]
    hist_cols = [
        f"hist_{channel}_{idx}"
        for channel in ("r", "g", "b")
        for idx in range(config.hist_bins)
    ]
    hue_bucket_cols = [
        "hue_red",
        "hue_orange",
        "hue_yellow",
        "hue_green",
        "hue_cyan",
        "hue_blue",
        "hue_indigo",
        "hue_violet",
    ]
    return rgb_mean_cols + rgb_std_cols + hsv_mean_cols + hist_cols + hue_bucket_cols


FEATURE_COLUMNS = build_feature_columns()


def rgb_to_hsv(arr: np.ndarray) -> np.ndarray:
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc
    s = np.divide(delta, maxc, out=np.zeros_like(delta), where=maxc > 0)

    safe_delta = np.where(delta > 0, delta, 1.0)
    h_r = ((g - b) / safe_delta) % 6
    h_g = (b - r) / safe_delta + 2
    h_b = (r - g) / safe_delta + 4
    h = np.where(
        delta > 0,
        np.where(maxc == r, h_r, np.where(maxc == g, h_g, h_b)),
        0.0,
    ) / 6.0 % 1.0

    return np.stack([h, s, v], axis=-1)


def _ensure_pil_image(image_or_path: Image.Image | str | Path) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        return image_or_path
    return Image.open(image_or_path).convert("RGB")


def extract_feature_vector(
    image_or_path: Image.Image | str | Path,
    config: ImageFeatureConfig = DEFAULT_IMAGE_CONFIG,
) -> list[float]:
    img = _ensure_pil_image(image_or_path)
    img = img.convert("RGB").resize(config.image_size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    rgb_means = [float(r.mean()), float(g.mean()), float(b.mean())]
    rgb_stds = [float(r.std()), float(g.std()), float(b.std())]

    hsv = rgb_to_hsv(arr)
    hsv_means = hsv.mean(axis=(0, 1)).tolist()

    n_px = config.image_size[0] * config.image_size[1]
    hists = np.concatenate(
        [
            np.histogram(channel.ravel(), bins=config.hist_bins, range=(0.0, 1.0))[0]
            for channel in (r, g, b)
        ]
    ).astype(np.float32) / n_px

    sat_mask = hsv[..., 1] > config.saturation_threshold
    if sat_mask.sum() > 0:
        hue_buckets = (
            np.histogram(hsv[..., 0][sat_mask], bins=8, range=(0.0, 1.0))[0]
            .astype(np.float32)
            / sat_mask.sum()
        )
    else:
        hue_buckets = np.zeros(8, dtype=np.float32)

    return rgb_means + rgb_stds + hsv_means + hists.tolist() + hue_buckets.tolist()


def extract_feature_frame(
    images: list[Image.Image | str | Path],
    config: ImageFeatureConfig = DEFAULT_IMAGE_CONFIG,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    cols = feature_columns or build_feature_columns(config)
    rows = [extract_feature_vector(image, config=config) for image in images]
    return pd.DataFrame(rows, columns=cols)

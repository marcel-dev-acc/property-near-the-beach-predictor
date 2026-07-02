"""
Rightmove listing scraper — mirrors notebooks/06_predict.ipynb strategy.

Fetches a listing page, extracts image URLs from the embedded ``propertyData``
JSON (with an ``<img>`` fallback), and downloads them into memory.
"""
from __future__ import annotations

import io
import json
import re

import requests
from PIL import Image

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
_TIMEOUT = 15


def _extract_image_urls(html: str) -> list[str]:
    """Pull image URLs from Rightmove's embedded JSON, falling back to <img> tags."""
    urls: list[str] = []

    # Primary: URLs inside the embedded property JSON (…/max_…jpeg etc.)
    urls = re.findall(r'https://media[^"\\ ]+?\.(?:jpe?g|png)', html)

    # Secondary: try the propertyData JSON blob explicitly
    if not urls:
        m = re.search(r'window\.PAGE_MODEL\s*=\s*(\{.*?\})\s*</script>', html, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                images = data.get("propertyData", {}).get("images", [])
                urls = [img["url"] for img in images if img.get("url")]
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass

    # Fallback: <img data-src="…"> / <img src="…">
    if not urls:
        urls = re.findall(r'<img[^>]+data-src="([^"]+)"', html)
    if not urls:
        urls = re.findall(r'<img[^>]+src="(https://media[^"]+)"', html)

    # De-duplicate while preserving order
    seen: set[str] = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def fetch_listing_images(url: str, max_images: int = 20) -> list[Image.Image]:
    """Return a list of PIL images downloaded from a Rightmove listing URL."""
    resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    resp.raise_for_status()

    img_urls = _extract_image_urls(resp.text)[:max_images]

    images: list[Image.Image] = []
    for img_url in img_urls:
        try:
            r = requests.get(img_url, headers=_HEADERS, timeout=_TIMEOUT)
            r.raise_for_status()
            images.append(Image.open(io.BytesIO(r.content)).convert("RGB"))
        except Exception:
            continue  # skip broken / non-image responses
    return images

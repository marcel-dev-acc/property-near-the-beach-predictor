# Project Plan — Property Near The Beach Predictor

## Overview

Build a **binary image classifier** that predicts whether a property is a beachfront
property or not. The model is trained on labelled room/scene images, then deployed
in a scraper notebook that accepts a Rightmove listing URL, downloads the listing's
photos, and predicts whether any image shows a beach view.

---

## Problem Statement

| | |
|---|---|
| **Task** | Binary image classification |
| **Classes** | `1 = beach`, `0 = not_beach` |
| **Input at training** | JPEG images in `data/raw/beach/` and `data/raw/not_beach/` |
| **Input at prediction** | A Rightmove listing URL |
| **Output** | Per-image prediction + overall verdict for the property |

---

## Data

### Raw Data — `data/raw/`

| Folder | Description | Example files |
|---|---|---|
| `beach/` | Exterior beach / coastal scenes | `i0001.jpg`, `k0001.jpg`, `s0001.jpg` |
| `not_beach/` | Interior rooms (bathroom, bedroom, dining, kitchen, living) | `bath_1.jpg`, `bed_1.jpg`, `kitchen_1.jpg` |

> **Why interior rooms as the negative class?**  
> Rightmove listings are dominated by indoor room photos. A beachfront property
> is flagged if *any* of its images shows a coastal scene — the rest will be
> indoors. This makes the class boundary natural and realistic.

### Interim Data — `data/interim/`

| File | Description |
|---|---|
| `image_manifest.parquet` | One row per image: `filepath`, `label`, `split` |
| `features.parquet` | Flat feature matrix extracted from all images |

### Processed Data — `data/processed/`

| File | Description |
|---|---|
| `train.parquet` | 80 % stratified training split |
| `test.parquet` | 20 % held-out test split |

---

## Pipeline — Notebooks in Order

```
data/raw/beach/          data/raw/not_beach/
        │
        ▼  01_extract  (build manifest, validate)
        │
data/interim/image_manifest.parquet
        │
        ▼  02_transform  (load images → extract features)
        │
data/interim/features.parquet
        │
        ▼  03_split  (stratified train / test split)
        │
data/processed/{train,test}.parquet
        │
        ▼  04_eda  (class balance, colour distributions, sample grids)
        │
        ▼  05_model  (train classifier, evaluate, save model)
        │
models/beach_predictor.joblib
        │
        ▼  06_predict  (scrape Rightmove URL → classify images)
        │
Verdict: beach / not beach
```

---

## Notebook Responsibilities

### 01 — Extract

**Goal:** Build an image manifest; perform zero-transformation validation.

Steps:
1. Walk `data/raw/beach/` and `data/raw/not_beach/` with `pathlib.Path.rglob("*.jpg")`
2. Assign labels: `beach → 1`, `not_beach → 0`
3. Record: `filepath` (relative), `filename`, `label`, `class_name`
4. Validate: assert both classes are non-empty, check files are readable
5. Print class counts and a random sample
6. Save manifest → `data/interim/image_manifest.parquet`

### 02 — Transform

**Goal:** Convert raw images to a flat, model-ready feature DataFrame.

Feature extraction strategy (pure `Pillow` + `NumPy` + `scikit-learn`, no deep learning):

| Feature group | Description | # of features |
|---|---|---|
| RGB channel means | Mean R, G, B across all pixels | 3 |
| RGB channel stds | Std R, G, B across all pixels | 3 |
| HSV channel means | Mean H, S, V (hue captures ocean blue / sandy gold) | 3 |
| RGB histogram | 32 bins per channel, flattened | 96 |
| Dominant hue bucket | Proportion of pixels in 8 hue buckets (red/orange/yellow/green/cyan/blue/indigo/violet) | 8 |
| **Total** | | **113** |

> Beach images are strongly characterised by blue/cyan hues (water, sky) and
> sandy golden hues. Interior rooms have more neutral/grey/brown palettes. This
> makes colour histograms a powerful discriminating feature without needing a
> neural network.

Steps:
1. Load manifest from `data/interim/image_manifest.parquet`
2. For each image: open with `Pillow`, resize to **128 × 128**, extract features
3. Assemble into a DataFrame; add `label` column
4. Handle corrupt/unreadable files gracefully (log and skip)
5. Save → `data/interim/features.parquet`

### 03 — Split

**Goal:** Create reproducible, stratified train / test splits.

Steps:
1. Load `data/interim/features.parquet`
2. Stratified split: **80 % train / 20 % test** (preserve class balance)
3. Print class distribution in each split
4. Save `data/processed/train.parquet` and `data/processed/test.parquet`

### 04 — EDA (Exploratory Data Analysis)

**Goal:** Understand the data visually and statistically before modelling.

Analyses:
1. **Class balance bar chart** — how many beach vs not_beach images
2. **Sample image grid** — 5 × 2 random samples per class
3. **Average image per class** — pixel-mean image for beach and not_beach
4. **Colour histogram overlay** — R, G, B distributions compared across classes
5. **HSV hue distribution** — hue histogram to show blue dominance in beach class
6. **Feature correlation heatmap** — top 20 correlated features with label
7. **Statistical test** — independent t-test on mean blue channel: beach vs not_beach (with p-value)

### 05 — Model Training & Evaluation

**Goal:** Train, compare, and persist the best classifier.

Models to compare:

| Model | Justification |
|---|---|
| **Logistic Regression** (baseline) | Fast, interpretable coefficients |
| **Random Forest** | Handles non-linear colour relationships; robust to noise |
| **Support Vector Machine (RBF kernel)** | Strong on high-dimensional feature spaces |

Steps:
1. Load `data/processed/train.parquet` and `data/processed/test.parquet`
2. Scale features with `StandardScaler` (important for LR and SVM)
3. Train all three models; compare with cross-validation (5-fold)
4. Evaluate on test set: accuracy, precision, recall, F1, ROC-AUC
5. Plot confusion matrix and ROC curve for the best model
6. Save the best model + scaler as a pipeline → `models/beach_predictor.joblib`

### 06 — Predict (Rightmove Scraper)

**Goal:** Accept a Rightmove listing URL, extract images, classify each, and return a verdict.

Scraping strategy:
1. Use `requests` with a browser-like `User-Agent` header to fetch the listing HTML
2. Parse the page with `BeautifulSoup`
3. Extract image URLs from Rightmove's embedded JSON (present in a `<script>` tag as `propertyData`) using `json` + `re`
4. Fall back to scraping `<img>` tags with `data-src` attributes if JSON extraction fails
5. Download each image into memory (`io.BytesIO`) — no disk writes required
6. Run each image through the **same feature extraction pipeline** as training (resize 128×128, extract 113 features)
7. Load the saved `beach_predictor.joblib` pipeline and predict per image
8. Display a grid of images with predicted labels overlaid

Prediction logic:

```
if any(image_predictions == 1):
    verdict = "🏖️  BEACH FRONT — this property appears to have beach views"
else:
    verdict = "🏠  NOT BEACH FRONT — no coastal images detected"
```

---

## Dependencies to Add

The current `requirements.txt` covers core data tools. Add:

```
Pillow>=10.0           # image loading and resizing
beautifulsoup4>=4.12   # HTML parsing for scraper
requests>=2.31         # HTTP for scraper
seaborn>=0.13          # EDA plots
```

---

## src/config.py

Create a shared configuration module at `src/config.py`:

```python
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[1]
RAW_DIR      = ROOT / "data" / "raw"
BEACH_DIR    = RAW_DIR / "beach"
NOT_BEACH_DIR = RAW_DIR / "not_beach"
INTERIM_DIR  = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR   = ROOT / "models"

MANIFEST_FILE = INTERIM_DIR / "image_manifest.parquet"
FEATURES_FILE = INTERIM_DIR / "features.parquet"
TRAIN_FILE    = PROCESSED_DIR / "train.parquet"
TEST_FILE     = PROCESSED_DIR / "test.parquet"
MODEL_FILE    = MODELS_DIR / "beach_predictor.joblib"

IMAGE_SIZE    = (128, 128)   # resize target for all images
HIST_BINS     = 32           # bins per channel in colour histogram
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
TARGET_COLUMN = "label"

def ensure_dirs():
    for d in [INTERIM_DIR, PROCESSED_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
```

---

## File Structure (end state)

```
property-near-the-beach-predictor/
├── data/
│   ├── raw/
│   │   ├── beach/           ← source beach images (read-only)
│   │   └── not_beach/       ← source non-beach images (read-only)
│   ├── interim/
│   │   ├── image_manifest.parquet
│   │   └── features.parquet
│   └── processed/
│       ├── train.parquet
│       └── test.parquet
├── models/
│   └── beach_predictor.joblib
├── notebooks/
│   ├── 01_extract.ipynb
│   ├── 02_transform.ipynb
│   ├── 03_split.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_model.ipynb
│   └── 06_predict.ipynb     ← new: Rightmove scraper + classifier
├── src/
│   ├── __init__.py
│   └── config.py
├── requirements.txt
├── README.md
└── PLAN.md
```

---

## Success Criteria

| Metric | Target |
|---|---|
| Test set accuracy | ≥ 90 % |
| Test set F1 (beach class) | ≥ 0.88 |
| ROC-AUC | ≥ 0.95 |
| Rightmove scraper | Returns a verdict within 10 seconds |

---

## Methodology Justification

Classical ML (scikit-learn) is chosen over deep learning because:

1. **Dataset size** — a few thousand images is small by deep learning standards; classical models on handcrafted colour features perform well and generalise reliably
2. **Interpretability** — colour histograms are human-understandable features; feature importances from Random Forest can be explained
3. **No GPU required** — the entire pipeline runs on a standard laptop
4. **Assessment alignment** — the project uses scikit-learn (LO1.3, LO4.1) and follows a systematic ETL structure (LO2.1)

The colour histogram approach is justified because the beach/not_beach boundary is
strongly correlated with colour distribution — coastal scenes contain significantly
more blue and cyan pixels than interior rooms, making this a learnable signal with
high signal-to-noise ratio.

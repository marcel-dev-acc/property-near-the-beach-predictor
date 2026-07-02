# Property Near The Beach Predictor

[INTRODUCTION HERE]

Beach Data Source: https://www.kaggle.com/datasets/irvingvasquez/publicsargazods

<i>I used all the images from this dataset</i>

Non-Beach Data Source: https://www.kaggle.com/datasets/mikhailma/house-rooms-streets-image-dataset

<i>I only used the house images from this dataset</i>


## Table of Contents
1. Description of the key principles of statistics, probability, and data analysis
2. Business Case Description & Outline (Hypothesis definition / testing)
3. Justify the selection of research methodologies
4. Implementation
5. Features
6. Model
7. Assessment of the effectiveness of the model
8. Tooling
9. Learning process and how the project has prepared me for adaptation in the field
10. Project Management


## 1. Description of the key principles of statistics, probability, and data analysis

### Statistics

Statistics provides the foundation for understanding and interpreting data. Key principles include:
- **Descriptive Statistics**: Summarizing data through measures like mean, median, standard deviation, and correlation to understand central tendency and variability.
- **Inferential Statistics**: Making generalizations about a population based on sample data, allowing us to test hypotheses about relationships in our dataset.
- **Distribution Analysis**: Understanding how data is distributed helps identify patterns, outliers, and assumptions for modeling (e.g., normality, skewness).
- **Hypothesis Testing**: Rigorously testing whether observed patterns are statistically significant or due to chance.

### Probability

Probability quantifies uncertainty and underlies all statistical inference:
- **Conditional Probability**: Understanding how the probability of an event changes given new information—essential for predictive modeling.
- **Bayes' Theorem**: Updating beliefs based on evidence; the foundation for probabilistic classification models.
- **Probability Distributions**: Different data types follow different distributions (binomial, normal, Poisson), which inform modeling choices.
- **Independence & Correlation**: Recognizing when variables are independent versus dependent, which affects feature selection and model assumptions.

### Data Analysis

Data analysis bridges statistics and actionable insights:
- **Exploratory Data Analysis (EDA)**: Understanding data structure, distributions, relationships, and anomalies before modeling.
- **Feature Engineering**: Creating meaningful variables that capture domain knowledge and improve model performance.
- **Data Quality Assessment**: Identifying and handling missing values, outliers, and inconsistencies to ensure reliable results.
- **Reproducibility**: Documenting the entire pipeline ensures results are verifiable and stakeholders trust the predictions.


## 3. Justify the selection of research methodologies 

[METHODOLOGIES]

### ETL flow

The notebooks are designed to run in order, each consuming the previous
stage's output:

```
data/raw/properties.csv
        │  01_extract  (validate)
        ▼
        │  02_transform
        ▼
data/interim/properties_transformed.parquet
        │  03_split
        ▼
data/processed/{train,test}.parquet
        │  04_eda → 05_model
        ▼
models/beach_predictor.joblib
```

[JUSTIFICATIONS]



## 7. Assessment of the effectiveness of the model

[METRICS]

[SCREENSHOTS]

[INTERPRETATION]


## 8. Tooling

[Juypter notebooks]

[Tableu]

[Streamlit]


## 9. Learning process and how the project has prepared me for adaptation in the field

[MY EXPERIENCES]


## Deployment

The trained model (`models/beach_predictor.joblib`) is served through a
[Streamlit](https://streamlit.io/) web app and deployed to
[Render](https://render.com/) using a Docker image. Render builds the
`Dockerfile` in the repository root, runs the container, and exposes it on a
public HTTPS URL.

### 1. Dockerfile

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the trained model
COPY . .

# Render sets the PORT env var; Streamlit must bind to it on 0.0.0.0
ENV PORT=8501
EXPOSE 8501

# Use shell form so $PORT is expanded at runtime
CMD streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true
```

> **Note:** Add `streamlit>=1.30` to `requirements.txt` so it is installed
> inside the image. Ensure `models/beach_predictor.joblib` is committed (or
> generated during the build) since `models/*` is currently git-ignored — Render
> only has access to what is in the repository.

### 2. `.dockerignore`

Keep the image small and the build fast by excluding data and virtual
environments:

```
.venv/
venv/
env/
.git/
.ipynb_checkpoints/
data/raw/
data/interim/
data/processed/
__pycache__/
*.pyc
```

### 3. Deploy on Render

1. Push the repository (including the `Dockerfile`, `app.py`, and the trained
   `.joblib` model) to GitHub.
2. Sign in to the [Render Dashboard](https://dashboard.render.com/) and click
   **New → Web Service**.
3. Connect your GitHub account and select this repository.
4. Configure the service:
   - **Language / Runtime:** `Docker` (Render auto-detects the `Dockerfile`).
   - **Region:** choose the one closest to your users.
   - **Branch:** `main`.
   - **Instance Type:** the **Free** plan is sufficient for a demo (note: free
     services spin down after inactivity and cold-start on the next request).
5. Render automatically injects a `PORT` environment variable — the Dockerfile
   already binds Streamlit to it, so no extra configuration is required. Add any
   additional environment variables under **Environment** if your app needs
   them.
6. Click **Create Web Service**. Render will build the Docker image, start the
   container, and publish the app at
   `https://<your-service-name>.onrender.com`.
7. Every push to the configured branch triggers an automatic rebuild and
   redeploy. You can also trigger a manual deploy from the dashboard.

### 4. Local test (optional)

Verify the container works before deploying:

```bash
docker build -t beach-predictor .
docker run -p 8501:8501 -e PORT=8501 beach-predictor
# open http://localhost:8501
```


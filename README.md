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


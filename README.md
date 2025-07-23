# Document-Clustering-for-Topic-Modelling

This repository explores **unsupervised machine learning techniques** to perform **topic modeling and document clustering** on a given corpus of text documents. The goal is to automatically discover abstract topics within documents and group similar ones together using clustering algorithms.

## Project Overview

This project involves:
- Preprocessing raw textual data (tokenization, stopword removal, stemming, etc.)
- Converting text into numerical format using TF-IDF
- Applying **K-Means Clustering** to group documents
- Using **Latent Dirichlet Allocation (LDA)** for topic modeling
- Visualizing clusters and topics using dimensionality reduction techniques (e.g., PCA, t-SNE)

## Technologies & Libraries Used

- Python 3.x
- Scikit-learn
- NLTK
- Gensim
- pandas, NumPy
- Matplotlib, Seaborn
- WordCloud
- spaCy

## Core Concepts

### Document Preprocessing
- Lowercasing, removing punctuation
- Stopword removal
- Lemmatization using spaCy or stemming via NLTK
- Tokenization

### Feature Extraction
- **TF-IDF Vectorization** to convert documents into numerical features.

### Clustering
- **K-Means Clustering** is used to group similar documents.
- Optimal number of clusters determined via **Elbow Method** and **Silhouette Score**.

### Topic Modeling
- **LDA (Latent Dirichlet Allocation)** to discover hidden thematic structure.

### Visualization
- **PCA** or **t-SNE** for dimensionality reduction
- Word clouds for topic interpretation

## Sample Results

- KMeans formed `k` coherent clusters with Silhouette Score: `X.XX`
- LDA identified topics like:
  - Topic 1: `data`, `model`, `learning`, `algorithm`
  - Topic 2: `health`, `disease`, `treatment`, `patient`

## Folder Structure
```plaintext
Document-Clustering-for-Topic-Modelling/
│
├── preprocessing.py # Text cleaning and preprocessing script
├── clustering.py # KMeans and evaluation code
├── topic_modeling.py # LDA topic modeling implementation
├── visualization.py # PCA and t-SNE visualizations
└── README.md # Project documentation



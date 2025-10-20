# Wine Clustering Analysis 

This repository implements an **unsupervised clustering analysis** on the **Kaggle Wine Dataset**.

It uses and compares two popular clustering algorithms: **K-Means**, **Gaussian Mixture Models (GMM)** and other famous algorithms. The project also leverages dimensionality reduction techniques like **PCA** and **t-SNE** to visualize the high-dimensional clusters in 2D space.

---

## 📘 Overview

The project demonstrates a complete unsupervised machine learning workflow. It begins with data loading and preprocessing (feature scaling), then moves to model selection by finding the optimal number of clusters using the **Elbow Method** and **Silhouette Score**.

Finally, it applies K-Means and GMM to segment the wine samples and visualizes the resulting clusters, providing a clear comparison of the two methods.

---

## 🧠 Problem Statement

Given a dataset of wines characterized by their chemical properties (e.g., 'Alcohol', 'Malic_Acid', 'Ash', 'Proline'), can we identify distinct, underlying groups or types of wine?

This is an **unsupervised learning** task to segment the data into clusters without any prior labels.

---

## ⚙️ Tech Stack

-   **Python 3.x**
-   **scikit-learn** (for K-Means, GMM, PCA, t-SNE, StandardScaler, and metrics)
-   **Pandas** & **NumPy** (for data loading and manipulation)
-   **Matplotlib** & **Seaborn** (for data visualization)
-   **Jupyter Notebook**

---

## 🚀 Features

-   **Data Preprocessing**: Feature scaling using `StandardScaler` to normalize the data.
-   **Optimal Cluster Selection**: Uses the **Elbow Method** and **Silhouette Score** to determine the best number of clusters (k) for K-Means.
-   **Clustering Models**: Implements and compares two algorithms:
    -   **K-Means Clustering** (a partitional clustering method).
    -   **Gaussian Mixture Models (GMM)** (a probabilistic, model-based clustering method).
-   **Dimensionality Reduction**: Employs **PCA** and **t-SNE** to reduce the 13 chemical features into 2 components for effective visualization.
-   **Rich Visualization**: Generates scatter plots to clearly show the discovered clusters.

---

## ▶️ How to Run

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Ramcharan2905/wine-clustering
    cd wine-clustering-analysis
    ```

2.  **Set up the environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Get the Data**
    Download the `wine-clustering.csv` dataset from the [Kaggle Wine Dataset](https://www.kaggle.com/datasets/fedesoriano/wine-clustering) page and place it in the root of the project directory.

5.  **Run notebooks**
    ```bash
    jupyter notebook
    ```
    Open `wine_clustering.ipynb` to run the analysis.

---

## 📊 Results

-   **Models**: K-Means, Gaussian Mixture Model
-   **Task**: Unsupervised Clustering
-   **Dataset**: Wine Dataset (Kaggle)
-   **Key Finding**: The analysis successfully segments the wine data into distinct clusters (found to be k=3), which align well with the known types of wine. The visualizations using PCA and t-SNE confirm the presence of three well-separated groups.

---

## 📈 Learning Purpose

This project aims to help practitioners understand:
-   How to implement an end-to-end unsupervised learning pipeline.
-   Techniques for selecting the optimal number of clusters (Elbow, Silhouette).
-   The implementation and comparison of partitional (K-Means) vs. probabilistic (GMM) clustering.
-   The importance of feature scaling for distance-based algorithms.
-   How to use PCA and t-SNE for visualizing high-dimensional data.

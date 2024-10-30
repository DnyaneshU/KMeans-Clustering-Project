# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_data():
    # Load the Iris dataset
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    return df

def preprocess_data(df):
    # Standardize features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled

def kmeans_clustering(df, n_clusters=3):
    # Fit KMeans with specified clusters
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(df)
    return kmeans

def plot_clusters(df, kmeans):
    # Apply PCA to reduce dimensions to 2 for visualization
    pca = PCA(2)
    df_pca = pca.fit_transform(df)

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")
    plt.title("K-Means Clustering on Iris Dataset")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Step 1: Load data
    df = load_data()

    # Step 2: Preprocess data
    df_scaled = preprocess_data(df)

    # Step 3: Cluster data
    kmeans = kmeans_clustering(df_scaled, n_clusters=3)

    # Step 4: Plot clusters
    plot_clusters(df_scaled, kmeans)

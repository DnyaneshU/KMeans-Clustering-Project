import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

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

def calculate_wcss(df):
    # Calculate WCSS for different values of k
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    return wcss

def plot_elbow(wcss):
    # Plot WCSS for each k value
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
    plt.title("Elbow Method for Optimal k")
    plt.show()

if __name__ == "__main__":
    # Step 1: Load and preprocess data
    df = load_data()
    df_scaled = preprocess_data(df)

    # Step 2: Calculate and plot WCSS
    wcss = calculate_wcss(df_scaled)
    plot_elbow(wcss)

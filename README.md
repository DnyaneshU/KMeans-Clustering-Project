 K-Means Clustering Project

This project applies the **K-Means Clustering** algorithm on the popular **Iris Dataset** to demonstrate how unsupervised learning can be used for clustering. To know more details about this algorithm, you can check out my detailed post on medium: 

## Project Structure

```
KMeans-Clustering-Project/
├── data/
│   └── iris.csv              # The dataset used for clustering
├── kmeans_clustering.py      # Main script for K-Means clustering
├── elbow_method.py           # Script for determining the optimal number of clusters using the Elbow Method
├── README.md                 # Project documentation
├── requirements.txt          # Dependencies required to run the project
```

## Explanation of Files
- **`kmeans_clustering.py`**: The primary script that loads the data, applies the K-Means clustering algorithm, and visualizes the results.
- **`elbow_method.py`**: Contains code to calculate Within-Cluster Sum of Squares (WCSS) and plot the Elbow Method graph to help determine the optimal number of clusters.
- **`requirements.txt`**: Lists the Python packages necessary to run the project.
- **`data/iris.csv`**: The dataset file (downloaded or provided) containing the Iris data for clustering.

## Dataset

We use the **Iris Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris). The Iris dataset is a classic dataset for clustering and classification, containing 150 samples of iris flowers with four features each: sepal length, sepal width, petal length, and petal width. The goal of clustering in this dataset is to group the samples into clusters that ideally correspond to different species of iris flowers.

### Why Iris Dataset?
The Iris dataset is commonly used in clustering projects because:
- It has a relatively small size, making it easy to work with.
- The features are well-suited for clustering into distinct groups.
- It's a standard dataset that enables easy comparison with other clustering algorithms and techniques.

## Getting Started

To set up this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DnyaneshU/KMeans-Clustering-Project.git
   cd KMeans-Clustering-Project
   ```

2. **Set up the virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Elbow Method script to determine optimal clusters**:
   ```bash
   python elbow_method.py
   ```

5. **Run the main clustering script**:
   ```bash
   python kmeans_clustering.py
   ```

## Results

- **Elbow Method:** The Elbow Method helps determine the optimal number of clusters by calculating the Within-Cluster Sum of Squares (WCSS) for different values of k. The "elbow" point in the graph represents the ideal number of clusters where adding more clusters doesn't significantly reduce WCSS, indicating an optimal grouping.

- **K-Means Clustering Visualization:** Once the optimal k value is selected, we apply the K-Means algorithm to cluster the iris samples. The final clusters are visualized on a scatter plot, where each color represents a different cluster, providing a clear separation of data points based on their similarities.

- **Centroid Calculation:** The algorithm also calculates centroids for each cluster, which are displayed as large points on the plot. These centroids represent the "center" of each cluster, helping to interpret the central tendency of each group.

- **Cluster Analysis:** By comparing the clustered data with the actual species labels (if available), we can analyze the effectiveness of K-Means in identifying meaningful groupings. This evaluation demonstrates how well unsupervised learning can approximate natural groupings within the dataset.

## Contributing

If you want to contribute to this project, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.


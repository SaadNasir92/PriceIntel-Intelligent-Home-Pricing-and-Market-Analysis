import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.data_preprocessing import DataPreprocessor
import os
import joblib


class KMeansClustering:
    def __init__(self, sample_size=50000, n_clusters=5, random_state=42):
        self.sample_size = sample_size
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.pca = None
        self.preprocessor = DataPreprocessor()
        self.create_output_dirs()

    def create_output_dirs(self):
        os.makedirs("outputs/cluster", exist_ok=True)
        os.makedirs("outputs/visualization_data/clustering_data", exist_ok=True)
        os.makedirs("models", exist_ok=True)

    def load_and_preprocess_data(self, filepath):
        X, _, _, _, self.feature_names = self.preprocessor.preprocess_data(
            filepath, target_column="price", sample_size=self.sample_size
        )
        return X

    def perform_clustering(self, X):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X)

    def elbow_method(self, X, max_clusters=10):
        inertias = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_clusters + 1), inertias, marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.title("Elbow Method for Optimal k")
        plt.savefig("outputs/cluster/elbow_method.png")
        plt.close()

        self.save_visualization_data(
            {"n_clusters": range(1, max_clusters + 1), "inertia": inertias},
            "elbow_method_data",
        )

    def visualize_clusters(self, X):
        self.pca = PCA(n_components=2)
        X_pca = self.pca.fit_transform(X)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1], c=self.kmeans.labels_, cmap="viridis"
        )
        plt.title("Cluster Visualization")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.colorbar(scatter)
        plt.savefig("outputs/cluster/cluster_visualization.png")
        plt.close()

        self.save_visualization_data(
            {"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Cluster": self.kmeans.labels_},
            "cluster_visualization_data",
        )

    def analyze_clusters(self, X):
        cluster_centers = self.kmeans.cluster_centers_
        cluster_sizes = np.bincount(self.kmeans.labels_)

        analysis_results = []
        for i, center in enumerate(cluster_centers):
            cluster_dict = {
                "Cluster": i,
                "Size": cluster_sizes[i],
                "Percentage": cluster_sizes[i] / len(X) * 100,
            }
            for j, feature in enumerate(self.feature_names):
                cluster_dict[feature] = center[j]
            analysis_results.append(cluster_dict)

        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv("outputs/cluster/cluster_analysis.csv", index=False)

        self.save_visualization_data(analysis_results, "cluster_analysis_data")

        return analysis_df

    def save_model(self):
        joblib.dump(self.kmeans, f"models/kmeans_clusters_{self.n_clusters}.joblib")

    def save_visualization_data(self, data, filename):
        pd.DataFrame(data).to_csv(
            f"outputs/visualization_data/clustering_data/{filename}.csv", index=False
        )

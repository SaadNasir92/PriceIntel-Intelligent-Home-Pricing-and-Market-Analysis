import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.metrics import silhouette_score
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
        self.selected_features = None

    def create_output_dirs(self):
        os.makedirs("outputs/cluster", exist_ok=True)
        os.makedirs("outputs/visualization_data/clustering_data", exist_ok=True)
        os.makedirs("models", exist_ok=True)

    def load_and_preprocess_data(self, filepath):
        X, _, y, _, self.feature_names = self.preprocessor.preprocess_data(
            filepath, target_column="price", sample_size=self.sample_size
        )
        return X, y

    def perform_clustering(self, X):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X)
        return self.kmeans.labels_

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
        if self.kmeans is None:
            raise ValueError(
                "Clustering has not been performed yet. Call perform_clustering first."
            )

        cluster_centers = self.kmeans.cluster_centers_
        cluster_sizes = np.bincount(self.kmeans.labels_)

        print(f"Shape of X: {X.shape}")
        print(f"Number of clusters: {self.n_clusters}")
        print(f"Shape of cluster_centers: {cluster_centers.shape}")
        print(f"Number of selected features: {len(self.selected_features)}")

        analysis_results = []
        for i, center in enumerate(cluster_centers):
            cluster_dict = {
                "Cluster": i,
                "Size": cluster_sizes[i],
                "Percentage": cluster_sizes[i] / len(X) * 100,
            }
            for j, feature in enumerate(self.selected_features):
                cluster_dict[feature] = center[j]
            analysis_results.append(cluster_dict)

        analysis_df = pd.DataFrame(analysis_results)
        analysis_df.to_csv("outputs/cluster/cluster_analysis.csv", index=False)

        return analysis_df

    def save_model(self):
        joblib.dump(self.kmeans, f"models/kmeans_clusters_{self.n_clusters}.joblib")

    def save_visualization_data(self, data, filename):
        pd.DataFrame(data).to_csv(
            f"outputs/visualization_data/clustering_data/{filename}.csv", index=False
        )

    def select_features(self, X, y, n_features=30):
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")
        print(f"Columns in X: {X.columns.tolist()[:10]}...")  # Print first 10 columns
        print(f"Data types in X:\n{X.dtypes.value_counts()}")

        # Aggregate closing date features
        closing_date_columns = [
            col for col in X.columns if col.startswith("closing_date_")
        ]
        if closing_date_columns:
            X["closing_month"] = (
                X[closing_date_columns]
                .idxmax(axis=1)
                .apply(lambda x: x.split("_")[2][:7])
            )
            X = X.drop(columns=closing_date_columns)

            # Convert closing_month to numerical
            try:
                X["closing_month"] = (
                    pd.to_datetime(X["closing_month"]).astype("int64") // 10**9
                )
                print("Successfully converted closing_month to int64 timestamp.")
            except Exception as e:
                print(f"Error converting closing_month: {str(e)}")
                print("Dropping closing_month column.")
                X = X.drop(columns=["closing_month"])

        # Identify potential categorical columns (one-hot encoded)
        categorical_mask = X.apply(lambda x: set(x.unique()).issubset({0, 1})).values
        categorical_columns = X.columns[categorical_mask].tolist()

        # Identify numerical columns (exclude one-hot encoded columns)
        numerical_columns = X.columns.difference(categorical_columns).tolist()

        print(f"Potential categorical columns: {len(categorical_columns)}")
        print(f"Numerical columns: {len(numerical_columns)}")

        # Ensure all numerical columns are float
        for col in numerical_columns:
            X[col] = X[col].astype(float)

        selected_features = []

        # Select top categorical features
        if categorical_columns:
            mi_scores = mutual_info_regression(X[categorical_columns], y)
            top_categorical = [
                categorical_columns[i]
                for i in np.argsort(mi_scores)[::-1][: n_features // 2]
            ]
            selected_features.extend(top_categorical)

        # Select top numerical features
        if numerical_columns:
            f_scores, _ = f_regression(X[numerical_columns], y)
            top_numerical = [
                numerical_columns[i]
                for i in np.argsort(f_scores)[::-1][: n_features // 2]
            ]
            selected_features.extend(top_numerical)

        self.selected_features = selected_features
        print(f"Selected features: {self.selected_features}")

        if len(self.selected_features) == 0:
            raise ValueError(
                "No features were selected. Check your data preprocessing steps."
            )

        print(
            f"Feature selection complete. Number of selected features: {len(self.selected_features)}"
        )
        print(f"Selected features: {self.selected_features}")

        return X[self.selected_features]

    def silhouette_analysis(self, X):
        silhouette_avg = silhouette_score(X, self.kmeans.labels_)
        return silhouette_avg

    # def hierarchical_clustering(self, X):
    #     hc = AgglomerativeClustering(n_clusters=self.n_clusters)
    #     hc_labels = hc.fit_predict(X)
    #     return hc_labels

    # def compare_clusterings(self, X, kmeans_labels, hc_labels):
    #     kmeans_silhouette = silhouette_score(X, kmeans_labels)
    #     hc_silhouette = silhouette_score(X, hc_labels)

    #     plt.figure(figsize=(12, 6))
    #     plt.bar(["K-means", "Hierarchical"], [kmeans_silhouette, hc_silhouette])
    #     plt.title("Silhouette Score Comparison")
    #     plt.ylabel("Silhouette Score")
    #     plt.savefig("outputs/cluster/clustering_comparison.png")
    #     plt.close()

    #     return {"kmeans_silhouette": kmeans_silhouette, "hc_silhouette": hc_silhouette}

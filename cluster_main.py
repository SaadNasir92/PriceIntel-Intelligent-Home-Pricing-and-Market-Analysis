from src.clustering import KMeansClustering
import time
import traceback
import psutil
import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

start_time = time.time()
process = psutil.Process()

try:
    clusterer = KMeansClustering(sample_size=250000, n_clusters=5)
    X, y = clusterer.load_and_preprocess_data(
        "data/processed/processed_synthetic_cleaned.csv"
    )

    logging.info(f"Initial shape of X: {X.shape}")
    logging.info(f"Initial columns: {X.columns.tolist()[:10]}...")
    logging.info(f"Initial data types:\n{X.dtypes.value_counts()}")

    # Feature streamlining
    X_selected = clusterer.select_features(X, y)
    logging.info(f"Shape after feature selection: {X_selected.shape}")
    logging.info(f"Number of selected features: {len(clusterer.selected_features)}")
    logging.info(f"Selected features: {clusterer.selected_features}")
    logging.info(
        f"Data types after feature selection:\n{X_selected.dtypes.value_counts()}"
    )

    if X_selected.shape[1] == 0:
        raise ValueError("No features were selected, cannot proceed with clustering")

    # Check for any remaining non-numeric data
    non_numeric = X_selected.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        logging.warning(f"Non-numeric data found in columns: {non_numeric}")
        logging.info("Converting to numeric...")
        X_selected = X_selected.apply(pd.to_numeric, errors="coerce")
        logging.info(
            f"Data types after conversion:\n{X_selected.dtypes.value_counts()}"
        )

    # Check for NaN values
    nan_columns = X_selected.columns[X_selected.isna().any()].tolist()
    if nan_columns:
        logging.warning(f"NaN values found in columns: {nan_columns}")
        logging.info("Dropping rows with NaN values...")
        X_selected = X_selected.dropna()
        y = y[X_selected.index]
        logging.info(f"Shape after dropping NaN values: {X_selected.shape}")

    # Perform clustering
    kmeans_labels = clusterer.perform_clustering(X_selected)

    # Validation and robustness check
    silhouette_avg = clusterer.silhouette_analysis(X_selected)
    logging.info(f"Silhouette Score: {silhouette_avg}")

    # hc_labels = clusterer.hierarchical_clustering(X_selected)
    # comparison_results = clusterer.compare_clusterings(
    #     X_selected, kmeans_labels, hc_labels
    # )
    # logging.info(f"Clustering Comparison Results: {comparison_results}")

    clusterer.elbow_method(X_selected)
    clusterer.visualize_clusters(X_selected)
    analysis_results = clusterer.analyze_clusters(X_selected)

    logging.info("Analysis results:")
    logging.info(analysis_results.head())
    analysis_results.to_csv("outputs/cluster/analysis_results.csv", index=False)
    clusterer.save_model()

except Exception as e:
    logging.error(f"An error occurred: {str(e)}")
    logging.error("Traceback:")
    traceback.print_exc()

finally:
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    logging.info(
        f"Peak memory usage: {process.memory_info().peak_wset / (1024 * 1024):.2f} MB"
    )

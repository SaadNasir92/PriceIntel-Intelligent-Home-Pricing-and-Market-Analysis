from src.clustering import KMeansClustering
import time
import psutil

start_time = time.time()
process = psutil.Process()


clusterer = KMeansClustering(sample_size=5000, n_clusters=5)
X = clusterer.load_and_preprocess_data("data/processed/processed_synthetic_cleaned.csv")
clusterer.perform_clustering(X)
clusterer.elbow_method(X)
clusterer.visualize_clusters(X)
analysis_results = clusterer.analyze_clusters(X)
print(analysis_results)
analysis_results.to_csv("outputs/cluster/analysis_results.csv", index=False)
clusterer.save_model()

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
print(f"Peak memory usage: {process.memory_info().peak_wset / (1024 * 1024):.2f} MB")

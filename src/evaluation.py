from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from joblib import parallel_backend
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


class ModelEvaluator:
    def __init__(self):
        self.plot_dir = "plots"
        os.makedirs(self.plot_dir, exist_ok=True)

    def evaluate_model(self, model, X_test, y_test):
        # Convert data to float32 if it's a KerasRegressor
        if "KerasRegressor" in str(type(model)):
            X_test = X_test.astype(np.float32)
            y_test = y_test.astype(np.float32)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

    def plot_predictions(self, model, X_test, y_test, max_points=10000):
        if len(y_test) > max_points:
            # Sample data points for plotting
            indices = np.random.choice(len(y_test), max_points, replace=False)
            X_test_sample = (
                X_test[indices]
                if isinstance(X_test, np.ndarray)
                else X_test.iloc[indices]
            )
            y_test_sample = (
                y_test[indices]
                if isinstance(y_test, np.ndarray)
                else y_test.iloc[indices]
            )
        else:
            X_test_sample = X_test
            y_test_sample = y_test

        y_pred = model.predict(X_test_sample)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_sample, y_pred, alpha=0.5)
        plt.plot(
            [y_test_sample.min(), y_test_sample.max()],
            [y_test_sample.min(), y_test_sample.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title("Actual vs Predicted Housing Prices (Sampled)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "actual_vs_predicted.png"))
        plt.close()

    def evaluate_multiple_models(self, models, X_test, y_test, X_train, y_train):
        results = {}
        for model_name, (model, cv_mean, cv_std) in models.items():
            print(f"Evaluating {model_name} model...")
            results[model_name] = {
                **self.evaluate_model(model, X_test, y_test),
                "CV_MSE_mean": cv_mean,
                "CV_MSE_std": cv_std,
            }
            self.plot_learning_curve(model, X_train, y_train, model_name)
        return results

    def compare_models_across_scalers(self, results):
        scalers = list(results.keys())
        models = list(results[scalers[0]]["metrics"].keys())
        metrics = ["MSE", "RMSE", "MAE", "R2"]

        for metric in metrics:
            plt.figure(figsize=(12, 6))
            for scaler in scalers:
                values = [results[scaler]["metrics"][model][metric] for model in models]
                plt.plot(models, values, marker="o", label=scaler)

            plt.title(f"{metric} Comparison Across Scalers")
            plt.xlabel("Models")
            plt.ylabel(metric)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f"{metric.lower()}_comparison.png"))
            plt.close()

    def find_best_model(self, results):
        best_rmse = float("inf")
        best_scaler = None
        best_model = None

        for scaler, scaler_results in results.items():
            for model_name, metrics in scaler_results["metrics"].items():
                if metrics["RMSE"] < best_rmse:
                    best_rmse = metrics["RMSE"]
                    best_scaler = scaler
                    best_model = scaler_results["models"][model_name]

        return best_scaler, best_model

    def plot_feature_importance(
        self, model, X, y, feature_names, model_name, n_top_features=20
    ):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
        else:
            result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
            importances = result.importances_mean
            indices = np.argsort(importances)[::-1]

        top_indices = indices[:n_top_features]
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {n_top_features} Feature Importances for {model_name}")
        plt.bar(range(n_top_features), importances[top_indices])
        plt.xticks(
            range(n_top_features), [feature_names[i] for i in top_indices], rotation=90
        )
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{model_name}_feature_importance.png"))
        plt.close()

    def plot_learning_curve(self, model, X, y, model_name):
        with parallel_backend("threading", n_jobs=2):
            train_sizes, train_scores, test_scores = learning_curve(
                model,
                X,
                y,
                cv=5,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring="neg_mean_squared_error",
            )
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)

        plt.figure()
        plt.title(f"Learning Curve for {model_name}")
        plt.xlabel("Training examples")
        plt.ylabel("Mean Squared Error")
        plt.plot(
            train_sizes, train_scores_mean, "o-", color="r", label="Training score"
        )
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="Cross-validation score",
        )
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{model_name}_learning_curve.png"))
        plt.close()

    def summarize_results(self, results):
        summary = {}
        for scaler, scaler_results in results.items():
            summary[scaler] = {
                model_name: {
                    "CV_MSE_mean": metrics["CV_MSE_mean"],
                    "CV_MSE_std": metrics["CV_MSE_std"],
                    "Test_RMSE": metrics["RMSE"],
                    "Test_R2": metrics["R2"],
                }
                for model_name, metrics in scaler_results["metrics"].items()
            }
        summary_df = pd.DataFrame(summary)
        return summary_df.to_csv("outputs/summary.csv", index=True)

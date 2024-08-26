from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os

class ModelEvaluator:
    def __init__(self):
        self.plot_dir = 'plots'
        os.makedirs(self.plot_dir, exist_ok=True)

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    def plot_predictions(self, model, X_test, y_test, max_points=10000):
        if len(y_test) > max_points:
            # Sample data points for plotting
            indices = np.random.choice(len(y_test), max_points, replace=False)
            X_test_sample = X_test[indices] if isinstance(X_test, np.ndarray) else X_test.iloc[indices]
            y_test_sample = y_test[indices] if isinstance(y_test, np.ndarray) else y_test.iloc[indices]
        else:
            X_test_sample = X_test
            y_test_sample = y_test

        y_pred = model.predict(X_test_sample)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_sample, y_pred, alpha=0.5)
        plt.plot([y_test_sample.min(), y_test_sample.max()], [y_test_sample.min(), y_test_sample.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Housing Prices (Sampled)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'actual_vs_predicted.png'))
        plt.close()

    def evaluate_multiple_models(self, models, X_test, y_test):
        results = {}
        for model_name, model in models.items():
            print(f"Evaluating {model_name} model...")
            results[model_name] = self.evaluate_model(model, X_test, y_test)
        return results

    def compare_models_across_scalers(self, results):
        scalers = list(results.keys())
        models = list(results[scalers[0]]['metrics'].keys())
        metrics = ['MSE', 'RMSE', 'MAE', 'R2']

        for metric in metrics:
            plt.figure(figsize=(12, 6))
            for scaler in scalers:
                values = [results[scaler]['metrics'][model][metric] for model in models]
                plt.plot(models, values, marker='o', label=scaler)
            
            plt.title(f'{metric} Comparison Across Scalers')
            plt.xlabel('Models')
            plt.ylabel(metric)
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f'{metric.lower()}_comparison.png'))
            plt.close()
            
    def find_best_model(self, results):
        best_rmse = float('inf')
        best_scaler = None
        best_model = None

        for scaler, scaler_results in results.items():
            for model_name, metrics in scaler_results['metrics'].items():
                if metrics['RMSE'] < best_rmse:
                    best_rmse = metrics['RMSE']
                    best_scaler = scaler
                    best_model = scaler_results['models'][model_name]

        return best_scaler, best_model
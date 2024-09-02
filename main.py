from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.utils import save_model
import matplotlib
import pandas as pd
import time
import psutil
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

matplotlib.use("Agg")
# configuring logger to stop the flood of font debugging messages
matplotlib.set_loglevel("WARNING")
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("matplotlib.pyplot").disabled = True


def main():
    start_time = time.time()
    process = psutil.Process()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    evaluator = ModelEvaluator()

    scaler_types = ["standard", "robust", "minmax"]
    results = {}

    sample_size = 50000  # Dataset size to be changed here
    feature_names = None

    for scaler_type in scaler_types:
        print(f"\nProcessing with {scaler_type} scaler:")

        # Preprocess data with all scalers
        X_train_scaled, X_test_scaled, y_train, y_test, current_feature_names = (
            preprocessor.preprocess_data(
                "data/processed/processed_synthetic_cleaned.csv",
                target_column="price",
                scaler_type=scaler_type,
                test_size=0.2,
                sample_size=sample_size,
            )
        )

        if feature_names is None:
            feature_names = current_feature_names

        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"Number of feature names: {len(feature_names)}")

        # Engineer features
        X_train_engineered, X_test_engineered, selected_features = (
            feature_engineer.engineer_features(
                X_train_scaled.to_numpy()
                if isinstance(X_train_scaled, pd.DataFrame)
                else X_train_scaled,
                X_test_scaled.to_numpy()
                if isinstance(X_test_scaled, pd.DataFrame)
                else X_test_scaled,
                y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train,
                feature_names,
            )
        )

        # Train and evaluate models
        trained_models = model_trainer.train_multiple_models(
            X_train_engineered, y_train, cv=5
        )
        scaler_results = evaluator.evaluate_multiple_models(
            trained_models, X_test_engineered, y_test, X_train_engineered, y_train
        )

        results[scaler_type] = {
            "metrics": scaler_results,
            "models": {k: v[0] for k, v in trained_models.items()},
            "feature_info": {
                "selected_feature_indices": feature_engineer.selected_feature_indices,
                "poly": feature_engineer.poly,
            },
        }

    # Compare models across all scalers
    evaluator.compare_models_across_scalers(results)

    # Find the best model across all scalers
    logger.debug("Starting best model selection")
    best_scaler, best_model = evaluator.find_best_model(results)
    logger.debug(
        f"Best model selected: {type(best_model).__name__} with {best_scaler} scaling"
    )
    print(
        f"\nBest overall model: {type(best_model).__name__} with {best_scaler} scaling"
    )

    logger.debug("Starting feature re-engineering for best model")
    # Re-scale the test data using the best scaler
    _, X_test_best_scaled, _, _, _ = preprocessor.preprocess_data(
        "data/processed/processed_synthetic_cleaned.csv",
        target_column="price",
        scaler_type=best_scaler,
        test_size=0.2,
        sample_size=sample_size,
    )

    # Re-engineer features for the best scaled test data using the stored feature engineer
    best_feature_info = results[best_scaler]["feature_info"]
    best_feature_engineer = FeatureEngineer()
    best_feature_engineer.selected_feature_indices = best_feature_info[
        "selected_feature_indices"
    ]
    best_feature_engineer.poly = best_feature_info["poly"]

    _, X_test_best_engineered, _ = best_feature_engineer.engineer_features(
        None,
        X_test_best_scaled.to_numpy()
        if isinstance(X_test_best_scaled, pd.DataFrame)
        else X_test_best_scaled,
        None,
        feature_names,
    )

    logger.debug("Feature re-engineering completed")
    # summary results to csv
    evaluator.summarize_results(results)

    # Plot feature importance for the best model (if it's RF or GB)
    if type(best_model).__name__ in [
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    ]:
        evaluator.plot_feature_importance(
            best_model,
            X_test_best_engineered,
            y_test,
            selected_features,
            f"{type(best_model).__name__}_{best_scaler}",
        )
    # saving best model
    if "KerasRegressor" in str(type(best_model)):
        print(f"Best model type: {type(best_model)}")
        print(f"Best model attributes: {dir(best_model)}")
        model_trainer.save_keras_model(
            best_model, f"models/best_model_{best_scaler}_KerasRegressor_{sample_size}"
        )
    else:
        save_model(
            best_model,
            f"best_model_{best_scaler}_{type(best_model).__name__}_{sample_size}",
        )

    logger.debug("Starting plot_predictions")
    # Plot predictions for the best model
    evaluator.plot_predictions(best_model, X_test_best_engineered, y_test)
    logger.debug("plot_predictions completed")

    """
    # Full dataset processing is commented out for now for testing

    print("\nProcessing full dataset with best model and scaler:")
    X_train_final, X_test_final, y_train_final, y_test_final, feature_names = preprocessor.preprocess_data(
        'data/processed/processed_synthetic_cleaned.csv',
        target_column='price',
        scaler_type=best_scaler,
        test_size=0.2
    )
    
    X_train_engineered_final, X_test_engineered_final, _ = feature_engineer.engineer_features(
        X_train_final.to_numpy() if isinstance(X_train_final, pd.DataFrame) else X_train_final,
        X_test_final.to_numpy() if isinstance(X_test_final, pd.DataFrame) else X_test_final,
        y_train_final,
        feature_names
    )
    best_model_final = model_trainer.train_model(X_train_engineered_final, y_train_final, model_type=best_model_name)

    # Save the best model trained on full dataset
    save_model(best_model_final, f'best_model_{best_scaler}_{best_model_name}_full')

    # Plot predictions for the best model on full dataset
    evaluator.plot_predictions(best_model_final, X_test_engineered_final, y_test_final)
    """

    end_time = time.time()
    # check how much time it takes per pipeline run
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    # check the max memory it user per pipeline run

    print(
        f"Peak memory usage: {process.memory_info().peak_wset / (1024 * 1024):.2f} MB"
    )


if __name__ == "__main__":
    main()

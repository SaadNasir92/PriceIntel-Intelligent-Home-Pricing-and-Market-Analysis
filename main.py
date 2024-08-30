from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.utils import save_model
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import time
import psutil

def main():
    start_time = time.time()
    process = psutil.Process()
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    evaluator = ModelEvaluator()

    scaler_types = ['standard', 'robust', 'minmax']
    results = {}

    sample_size =  200 # Dataset size to be changed here
    feature_names = None

    for scaler_type in scaler_types:
        print(f"\nProcessing with {scaler_type} scaler:")
        
        # Preprocess data with all scalers
        X_train_scaled, X_test_scaled, y_train, y_test, current_feature_names = preprocessor.preprocess_data(
            'data/processed/processed_synthetic_cleaned.csv',
            target_column='price',
            scaler_type=scaler_type,
            test_size=0.2,
            sample_size=sample_size
        )
        
        if feature_names is None:
            feature_names = current_feature_names

        print(f"X_train_scaled shape: {X_train_scaled.shape}")
        print(f"X_test_scaled shape: {X_test_scaled.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"Number of feature names: {len(feature_names)}")
            
        # Engineer features
        X_train_engineered, X_test_engineered, selected_features = feature_engineer.engineer_features(
            X_train_scaled.to_numpy() if isinstance(X_train_scaled, pd.DataFrame) else X_train_scaled,
            X_test_scaled.to_numpy() if isinstance(X_test_scaled, pd.DataFrame) else X_test_scaled,
            y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train,
            feature_names
        )

        # Train and evaluate models
        trained_models = model_trainer.train_multiple_models(X_train_engineered, y_train)
        scaler_results = evaluator.evaluate_multiple_models(trained_models, X_test_engineered, y_test, X_train_engineered, y_train)
        
        results[scaler_type] = {
            'metrics': scaler_results,
            'models': trained_models,
            'feature_engineer': feature_engineer
        }

    # Compare models across all scalers
    evaluator.compare_models_across_scalers(results)

    # Find the best model across all scalers
    best_scaler, best_model = evaluator.find_best_model(results)
    print(f"\nBest overall model: {type(best_model).__name__} with {best_scaler} scaling")
    
    # Re-scale the test data using the best scaler
    _, X_test_best_scaled, _, _, _ = preprocessor.preprocess_data(
    'data/processed/processed_synthetic_cleaned.csv',
    target_column='price',
    scaler_type=best_scaler,
    test_size=0.2,
    sample_size=sample_size)
    
    # Re-engineer features for the best scaled test data using the stored feature engineer
    best_feature_engineer = results[best_scaler]['feature_engineer']
    _, X_test_best_engineered, _ = best_feature_engineer.engineer_features(
        None,
        X_test_best_scaled.to_numpy() if isinstance(X_test_best_scaled, pd.DataFrame) else X_test_best_scaled,
        None, 
        feature_names
    )
    
    
    # Plot feature importance for the best model (if it's RF or GB)
    if type(best_model).__name__ in ['RandomForestRegressor', 'GradientBoostingRegressor']:
        evaluator.plot_feature_importance(
            best_model, 
            X_test_best_engineered, 
            y_test, 
            selected_features, 
            f"{type(best_model).__name__}_{best_scaler}"
        )
    
    # Save the best model
    save_model(best_model, f'best_model_{best_scaler}_{type(best_model).__name__}_{sample_size}')

    # Plot predictions for the best model
    evaluator.plot_predictions(best_model, X_test_best_engineered, y_test)
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
    print(f'Total execution time: {end_time - start_time:.2f} seconds')  # noqa: F821
    # check the max memory it user per pipeline run

    print(f"Peak memory usage: {process.memory_info().peak_wset / (1024 * 1024):.2f} MB")  # noqa: F821

if __name__ == "__main__":
    main()
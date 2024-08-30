# Housing Price Prediction Model: Next Steps and Model Selection

## Immediate Actions

1. **Fix model saving issue** - ✅ Completed
   - Modified `build_nn_model` in `ModelTrainer` class to use a regular function instead of a lambda to enable pickling.

2. **Implement additional features**
   - Add time tracking for training and evaluation of each model. - ✅ Completed
   - Implement memory usage tracking to monitor resource consumption. - ✅ Completed
   - Add feature importance plotting for tree-based models (Random Forest and Gradient Boosting). ✅ Completed
   - Implement learning curves to assess overfitting/underfitting for each model. ✅ Completed

3. **Optimize for larger datasets**
   - Implement batch processing for data loading and model training.
   - Use memory-efficient data types (e.g., float32 instead of float64) where appropriate.
   - Implement feature selection to reduce dimensionality before model training.

4. **Hyperparameter tuning**
   - Implement basic hyperparameter tuning for Ridge, Random Forest, and Gradient Boosting models.
   - Investigate optimal architecture and hyperparameters for Neural Network model to improve performance.

## Model Selection

Based on the results from our two 50,000 record runs, we've decided to focus on the following models:

### Models to Keep

1. **Ridge Regression**
   - Represents linear models with regularization.
   - Balanced performance among linear models.
   - Offers good interpretability.

2. **Random Forest**
   - Strong performance in both runs.
   - Provides feature importance and good interpretability.
   - Consistent performance across different scalers.

3. **Gradient Boosting**
   - Best performer in the second 50K run with MinMax scaling.
   - Offers feature importance and reasonable interpretability.
   - Shows potential for further improvement with hyperparameter tuning.

4. **Neural Network**
   - Performance varied between runs (best in first, underperformed in second).
   - Requires further investigation and optimization.
   - Keep for potential improvements with architecture adjustments and hyperparameter tuning.

### Models to Remove

1. **Support Vector Regression (SVR)**
   - Consistently underperformed across both 50K runs.
   - Highest RMSE, MAE, and MSE; lowest R2 score.
   - Remove due to poor performance and high computational cost.

2. **Linear Regression**
   - Outperformed by regularized versions (Ridge).
   - Remove in favor of Ridge Regression for better generalization.

3. **Lasso Regression**
   - Similar performance to Ridge, but with convergence issues in MinMax scaling.
   - Remove in favor of Ridge Regression for stability and performance.

## Performance Summary

- Gradient Boosting with MinMax scaling performed best in the latest run.
- Random Forest showed strong and consistent performance across runs.
- Ridge Regression performed well among linear models.
- Neural Network performance varied significantly between runs, requiring further investigation.
- MinMax and Robust scaling generally provided better results across models.

## Next Run Configuration

- Dataset size: 75,000 records (incrementally increasing from 50,000)
- Models: Ridge, Random Forest, Gradient Boosting, Neural Network
- Implement basic hyperparameter tuning for Ridge, RF, and GB
- Experiment with different Neural Network architectures to improve performance

## Resource Considerations

- Previous run (50,000 records):
  - Runtime: ~18.5 minutes
  - Peak memory usage: 3143.38 MB
  - Consider cloud computing options if resource usage increases significantly with 75,000 records

## Additional Information

- The full 250,000 record dataset may not be feasible on current hardware.
- Consider cloud computing options for full dataset if local resources are insufficient.
- Balance model performance with interpretability requirements for the interactive user interface.
- Implement cross-validation for more robust model evaluation.

## Next Steps

1. Implement hyperparameter tuning for Gradient Boosting and Random Forest models.
2. Investigate the drop in Neural Network performance and experiment with different architectures.
3. Analyze feature importance from the best-performing models (Gradient Boosting and Random Forest).
4. Implement the additional features listed in the Immediate Actions section.
5. Prepare for the 75,000 record run with the optimized models and configurations.
6. Continue monitoring performance and resource usage to ensure scalability.
# Housing Price Prediction Model: Next Steps and Model Selection

## Immediate Actions

1. **Fix model saving issue**
   - Modify `build_nn_model` in `ModelTrainer` class to use a regular function instead of a lambda to enable pickling.

2. **Implement additional features**
   - Add time tracking for training and evaluation of each model.
   - Implement memory usage tracking to monitor resource consumption.
   - Add feature importance plotting for tree-based models (Random Forest and Gradient Boosting).
   - Implement learning curves to assess overfitting/underfitting for each model.

3. **Optimize for larger datasets**
   - Implement batch processing for data loading and model training.
   - Use memory-efficient data types (e.g., float32 instead of float64) where appropriate.
   - Implement feature selection to reduce dimensionality before model training.

## Model Selection

Based on the results from our 50,000 record run, we've decided to focus on the following models:

### Models to Keep

1. **Ridge Regression**
   - Represents linear models with regularization.
   - Balanced performance among linear models.
   - Offers good interpretability.

2. **Random Forest**
   - Strong performance, second only to Neural Network.
   - Provides feature importance and good interpretability.
   - [Performance Graph](link_to_RMSE_comparison.png)

3. **Gradient Boosting**
   - Performance similar to Random Forest.
   - Offers feature importance and reasonable interpretability.
   - [Performance Graph](link_to_MAE_comparison.png)

4. **Neural Network**
   - Best performer across all metrics and scalers.
   - Requires further evaluation for interpretability and overfitting.
   - [Performance Graph](link_to_R2_comparison.png)

### Models to Remove

1. **Support Vector Regression (SVR)**
   - Significantly underperformed all other models.
   - Highest RMSE, MAE, and MSE; lowest R2 score.
   - [Performance Graph](link_to_MSE_comparison.png)

2. **Linear Regression**
   - Outperformed by regularized versions (Ridge and Lasso).
   - Removed in favor of Ridge Regression.

3. **Lasso Regression**
   - Similar performance to Ridge, but with convergence issues.
   - Removed in favor of Ridge Regression.

## Performance Summary

- Neural Network consistently outperformed other models across all scalers.
- Random Forest and Gradient Boosting showed strong performance, close to Neural Network.
- Ridge Regression performed best among linear models.
- Robust scaling generally provided slightly better results across models.

## Next Run Configuration

- Dataset size: 75,000 records (incrementally increasing from 50,000)
- Models: Ridge, Random Forest, Gradient Boosting, Neural Network
- Implement basic hyperparameter tuning for Ridge, RF, and GB
- Experiment with simpler Neural Network architecture

## Resource Considerations

- Previous run (50,000 records):
  - Runtime: ~22 minutes
  - Peak memory usage: 10.3/16GB
  - CPU utilization: 10%
- Monitor these metrics in the next run to assess scalability

## Additional Information

- The full 250,000 record dataset may not be feasible on current hardware.
- Consider cloud computing options for full dataset if local resources are insufficient.
- Balance model performance with interpretability requirements for the interactive user interface.
- Implement cross-validation for more robust model evaluation.
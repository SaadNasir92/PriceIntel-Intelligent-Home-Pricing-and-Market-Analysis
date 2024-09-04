# Housing Price Prediction Model: Next Steps and Model Selection

## Immediate Actions

1. **Fix model saving issue** - ✅ Completed
   - Modified `build_nn_model` in `ModelTrainer` class to use a regular function instead of a lambda to enable pickling.

2. **Implement additional features** ✅ Completed
   - Add time tracking for training and evaluation of each model. - 
   - Implement memory usage tracking to monitor resource consumption. - 
   - Add feature importance plotting for tree-based models (Random Forest and Gradient Boosting). 
   - Implement learning curves to assess overfitting/underfitting for each model. 

3. **Optimize for larger datasets**
   - Implement batch processing for data loading and model training.
   - Use memory-efficient data types (e.g., float32 instead of float64) where appropriate. ✅ Completed
   - Implement feature selection to reduce dimensionality before model training.

   ## Cross-Validation Implementation ✅ Completed

   ### Objective
   Implement cross-validation in the main pipeline to ensure more robust model evaluation and support hyperparameter tuning.

   ### Key Steps
   1. Modify `ModelTrainer` class to incorporate cross-validation:
      - Add a `cv_split` parameter to the `train_model` method.
      - Use `cross_val_score` or `cross_validate` from scikit-learn for model evaluation.
   2. Update `ModelEvaluator` class:
      - Modify `evaluate_model` method to handle cross-validation results.
      - Add a method to calculate and report average cross-validation scores.
   3. Adjust `main.py` to use cross-validation:
      - Add a parameter for number of CV folds (e.g., `n_cv_folds = 5`).
      - Pass CV parameter to model training and evaluation functions.

## 4. Hyperparameter Tuning

Based on the results from our fourth 50K run, where the KerasRegressor model with robust scaling showed the best performance, we will focus our hyperparameter tuning efforts as follows:

### Gradient Boosting ✅ Completed
# Hyperparameter Tuning for Gradient Boosting Regressor (GBR) with MinMax Scaling

## Model Selection ✅ Completed

# Decision to Simplify Model Pipeline - Update 9/3/24  ✅ Completed

## Reasons for Removing the Neural Network (NN) Model ✅ Completed

1. **Resource Intensive:**
   - The NN model requires significant computational resources, leading to prolonged training times and higher demands on system memory and processing power.

2. **Instability in Performance:**
   - The NN model has shown inconsistent learning curves with large fluctuations in error rates, indicating potential issues with overfitting or poor generalization.
   - These instabilities make the NN model less reliable and harder to tune effectively.

3. **Minimal Performance Gain:**
   - Despite the high resource investment, the NN model does not significantly outperform other models in terms of key performance metrics like RMSE, MSE, and R².
   - The performance gain does not justify the additional complexity and resource requirements.

## Reasons for Removing Ridge Regression and Random Forest Models ✅ Completed

1. **Suboptimal Performance:**
   - Both Ridge Regression and Random Forest models, while stable, do not outperform the Gradient Boosting model.
   - The difference in performance across key metrics (RMSE, MAE, MSE, R²) indicates that these models are less effective for the current task.

2. **Focus on the Best Model:**
   - By eliminating these models, we can concentrate our efforts on tuning and optimizing the best-performing model, which is Gradient Boosting.
   - This approach simplifies the pipeline and ensures that resources are focused where they are most impactful.

## Reasons for Removing Other Scalers ✅ Completed

1. **Inconsistent Results:**
   - The other scalers (e.g., RobustScaler, StandardScaler) have shown inconsistent performance, particularly in the Neural Network model where the RobustScaler significantly underperformed.
   - This inconsistency adds unnecessary variability to the results.

2. **MinMax Scaler Superiority:**
   - The MinMax scaler consistently produced better or comparable results across different models.
   - Standardizing on MinMax scaling reduces complexity in the preprocessing pipeline and ensures a more consistent input range for the model.

## Conclusion

By removing the three models (NN, Ridge, Random Forest) and other scalers, the machine learning pipeline is simplified and focused on the Gradient Boosting model with MinMax scaling. This decision is based on:

- The superior and consistent performance of the GB model.
- The resource efficiency gained by reducing the number of models and scalers.
- The elimination of unnecessary complexity, leading to a more manageable and effective pipeline.

This streamlined approach allows for more targeted hyperparameter tuning and better use of computational resources, ultimately improving the overall performance and reliability of the model.


### Model Evaluation 9/1/24 ✅ Completed
### Models to Keep 
This was based on the first couple runs. The information above is based on last run. Keras won. 
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
   - Fix the tensorflow error (guide to fix in tensorflowerror markdown file) ✅ Completed

### Models to Remove ✅ Completed

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

- GB Model: Across both sets, the GB model consistently showed good generalization and stability, making it the most reliable choice for your final pipeline.
- NN Model: The NN model displayed instability in both runs, with varying performance that was heavily influenced by the choice of scaler, suggesting it might not be the best   model to prioritize.
- Scalers: The MinMax scaler consistently outperformed others in both runs, making it the preferred choice for all models.



## Additional Information

- The full 250,000 record dataset may not be feasible on current hardware.
- Consider cloud computing options for full dataset if local resources are insufficient.
- Balance model performance with interpretability requirements for the interactive user interface.
- Implement cross-validation for more robust model evaluation.

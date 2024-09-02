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

### 4.1 KerasRegressor (Neural Network)

Primary focus due to best performance in previous run.
Addressing observed overfitting while maintaining model performance.


- **Architecture:**
  - Number of layers: [2, 3]
  - Neurons per layer: [32, 64, 128]
  - Activation functions: ['relu', 'elu']

- **Optimization:**
  - Learning rate: [0.01, 0.001, 0.0001]
  - Optimizer: ['adam']
  - Batch size: [32, 64, 128]
  - Epochs: Use early stopping with patience of 10 epochs

- **Regularization:**
  - Dropout rate: [0.1, 0.2, 0.3, 0.4, 0.5]
  - L2 regularization: [0.01, 0.001, 0.0001]

### 4.2 Gradient Boosting

Second priority for tuning due to strong performance.
Focus on controlling tree complexity to reduce overfitting.

- **Tree-specific:**
  - n_estimators: [100, 200, 500]
  - max_depth: [3, 4, 5, 6]
  - min_samples_split: [5, 10, 20]
  - min_samples_leaf: [2, 4, 8]

- **Boosting parameters:**
  - learning_rate: [0.01, 0.05, 0.1]
  - subsample: [0.6, 0.8]
  - colsample_bytree: [0.6, 0.8]

### 4.3 Random Forest

Include for comparison and potential ensemble methods.
Significant focus on reducing overfitting through tree complexity control.

- **Tree-specific:**
  - n_estimators: [100, 200]
  - max_depth: [5, 10, 15, None]
  - min_samples_split: [5, 10, 20]
  - min_samples_leaf: [2, 4, 8]

- **Randomness:**
  - max_features: ['sqrt', 'log2']

### 4.4 Ridge Regression

Maintain as baseline model with minimal tuning.
Explore slightly more complex models to address potential underfitting.


- alpha: [0.1, 0.5, 1.0, 5.0, 10.0]

### Implementation Strategy:

1. Use RandomizedSearchCV for efficient exploration of hyperparameter space.
2. Perform 5-fold cross-validation for each model.
3. Use 50 iterations for RandomizedSearchCV to balance exploration and computation time.
4. Use negative mean squared error (-MSE) as the scoring metric for consistency across models.
5. Retain the best-performing configuration for each model type.
6. For KerasRegressor, implement early stopping in model compilation:
   ```python
   from tensorflow.keras.callbacks import EarlyStopping
   early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
   ```
7. For tree-based models, use out-of-bag error estimates where applicable.

### Post-Tuning Analysis:

1. Compare tuned model performances against the baseline and each other.
2. Analyze learning curves to assess overfitting/underfitting and compare to previous results
3. Evaluate feature importance for tree-based models.
4. Consider ensemble methods combining top-performing models.
5. Analyze the selected hyperparameters to understand which had the most impact on reducing overfitting.
6. Perform a final evaluation on a held-out test set to confirm generalization capability.

### Monitoring during tuning:
1. Track both training and validation scores throughout the tuning process.
2. For each model, plot learning curves of the top 5 parameter combinations to visually inspect for overfitting/underfitting.
3. Monitor the difference between training and validation scores as a key indicator of overfitting.

## Model Selection ✅ Completed

Based on our most recent 50K run (9/1/24 - 10PM)

a) Best Overall Model: KerasRegressor (Neural Network) with robust scaling
   - Lowest RMSE: 70487.43
   - Highest R2 Score: 0.8383 (robust scaling)

b) Model Ranking (based on RMSE):
   1. Gradient Boosting (GB): 70525.85 (MinMax scaling)
   2. Ridge Regression: 70556.08 (Robust scaling)
   3. Neural Network (NN): 70487.43 (Robust scaling)
   4. Random Forest (RF): 71536.83 (MinMax scaling)

c) Scaling Impact:
   - MinMax scaling generally performed well across models
   - Robust scaling worked best for the Neural Network
   - Standard scaling showed consistent performance across models

2. Detailed Analysis:

a) Ridge Regression:
   - Consistent performance across all scalers
   - RMSE range: 70556.08 - 70576.73
   - R2 range: 0.8378 - 0.8380
   - Learning curve shows slight overfitting

b) Random Forest:
   - Worst performer among the models
   - RMSE range: 71536.83 - 72994.85
   - R2 range: 0.8266 - 0.8334
   - Learning curve shows significant overfitting

c) Gradient Boosting:
   - Best performer among traditional models
   - RMSE range: 70525.85 - 70547.01
   - R2 range: 0.8380 - 0.8381
   - Learning curve shows some overfitting, but less severe than Random Forest

d) Neural Network (KerasRegressor):
   - Best overall performance with robust scaling
   - Inconsistent performance across scalers
   - RMSE range: 70487.43 - 70789.09
   - R2 range: 0.8369 - 0.8383
   - Learning curve shows potential for improvement with more data or longer training

3. Observations and Recommendations:

a) Model Selection:
   - The Neural Network (KerasRegressor) with robust scaling shows the best performance and should be our primary focus for further optimization.
   - Gradient Boosting is a strong alternative and should be considered for ensemble methods or as a fallback option.

b) Scaling:
   - Different models perform best with different scalers. We should maintain separate pipelines for each model-scaler combination in future iterations.

c) Overfitting:
   - Random Forest and Gradient Boosting show signs of overfitting. Consider implementing regularization techniques or reducing model complexity.
   - The Neural Network's learning curve suggests it might benefit from more data or longer training.

d) Feature Engineering:
   - The current feature set (1495 initial features, 50 after engineering) seems effective but may be causing overfitting in tree-based models.
   - Consider feature selection techniques to reduce dimensionality.

e) Cross-Validation:
   - Our current 5-fold cross-validation approach is working well. Consider experimenting with different numbers of folds to ensure stability of results.

f) Hyperparameter Tuning:
   - Focus on tuning the Neural Network and Gradient Boosting models, as they show the most promise.
   - For Neural Network, prioritize regularization techniques to combat potential overfitting.
   - For Gradient Boosting, focus on tree complexity parameters to reduce overfitting while maintaining performance.

### Models to Keep ✅ Completed
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

- Previous run (50,000 records) AUG 29th 8PM:
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
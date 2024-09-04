## Hyperparameter Tuning Summary

We implemented a two-stage hyperparameter tuning process for our Gradient Boosting Regressor model using both GridSearchCV and RandomizedSearchCV from scikit-learn.

### Hyperparameters Tuned
- learning_rate
- n_estimators
- max_depth
- min_samples_leaf
- subsample
- max_features

### Tuning Methods
1. **Initial Grid Search**: Explored a predefined set of hyperparameter combinations.
2. **Subsequent Randomized Search**: Fine-tuned the model by exploring a wider range of values.

### Hyperparameter Ranges Explored

| Hyperparameter   | Grid Search        | Randomized Search           |
|------------------|---------------------|---------------------------|
| learning_rate    | [0.01, 0.05, 0.1]   | [0.01, 0.05, 0.1]         |
| n_estimators     | [100, 200, 300]     | [150, 200, 250, 300, 350] |
| max_depth        | [3, 5, 7]           | [3, 4, 5, 6, 7]           |
| min_samples_leaf | [1, 3, 5]           | [1, 2, 3, 4, 5]           |
| subsample        | [0.7, 0.85, 1.0]    | [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] |
| max_features     | [0.5, 0.75, 1.0]    | [0.3, 0.5, 0.7, 0.9, 1.0] |

### Evaluation Metric
- Negative Mean Squared Error (for consistency with scikit-learn's scoring system where higher is better)

### Cross-validation
- 5-fold cross-validation was used to ensure robust performance estimates.

This approach allowed us to systematically explore a wide range of hyperparameter combinations, balancing between a thorough search and computational efficiency. The two-stage process helped us to first identify promising regions in the hyperparameter space with Grid Search, and then refine our search with Randomized Search.

The final model's hyperparameters were selected based on the best cross-validation performance across all tested combinations.
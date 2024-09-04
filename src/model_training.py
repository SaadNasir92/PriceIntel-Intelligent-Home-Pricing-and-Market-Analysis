from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        pass

    def train_gbr_with_tuning(self, X_train, y_train):
        # Initial Grid Search
        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "min_samples_leaf": [1, 3, 5],
            "subsample": [0.7, 0.85, 1.0],
            "max_features": [0.5, 0.75, 1.0],
        }

        grid_search = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=12,
        )

        logger.info("Starting Grid Search for GBR...")
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters from Grid Search: {grid_search.best_params_}")

        # Fine-tuning with Randomized Search
        param_distributions = {
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [150, 200, 250, 300, 350],
            "max_depth": [3, 4, 5, 6, 7],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "max_features": [0.3, 0.5, 0.7, 0.9, 1.0],
        }

        random_search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_distributions,
            n_iter=50,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=12,
            random_state=42,
        )

        logger.info("Starting Randomized Search for GBR...")
        random_search.fit(X_train, y_train)
        logger.info(
            f"Best parameters from Randomized Search: {random_search.best_params_}"
        )

        return random_search.best_estimator_, random_search.cv_results_

    ###---------------------------------NOT USING METHODS BELOW ANYMORE SINCE WE ARE FOCUSED ON ONE SCALER AND ONE MODEL---------------------------------------------###

    # Not working when trying to save the model to a pickle file, approaching it differently below.
    ## Worked, keeping newer version (defining the model creation function outside the class)


#     def build_nn_model(self, input_dim):
#         return KerasRegressor(
#             model=create_model,
#             input_dim=input_dim,
#             epochs=100,
#             batch_size=32,
#             verbose=0,
#         )

#     def save_keras_model(self, model, filename):
#         print(f"Saving KerasRegressor model to {filename}")

#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(filename), exist_ok=True)

#         # Save the entire KerasRegressor object
#         joblib.dump(model, filename)
#         print(f"KerasRegressor model saved to {filename}")

#     def train_model(self, X_train, y_train, model_type="rf", params=None, cv=5):
#         if model_type not in self.models:
#             raise ValueError(f"Unsupported model type: {model_type}")

#         if isinstance(X_train, pd.DataFrame) or isinstance(X_train, pd.Series):
#             X_train = X_train.to_numpy()
#         if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
#             y_train = y_train.to_numpy()

#         if model_type == "nn":
#             model = self.build_nn_model(X_train.shape[1])
#             cv_scores = self.custom_cv_keras(X_train, y_train, cv=cv)
#         else:
#             model = self.models[model_type]

#         if params:
#             model = GridSearchCV(model, params, cv=5, scoring="neg_mean_squared_error")

#         cv_scores = cross_val_score(
#             model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
#         )

#         model.fit(X_train, y_train)

#         if isinstance(model, GridSearchCV):
#             print(f"Best parameters for {model_type}: {model.best_params_}")
#             return model.best_estimator_, -cv_scores.mean(), cv_scores.std()
#         return model, -cv_scores.mean(), cv_scores.std()

#     def train_multiple_models(self, X_train, y_train, cv=5):
#         trained_models = {}
#         for model_type in self.models.keys():
#             print(f"Training {model_type} model...")
#             trained_models[model_type] = self.train_model(
#                 X_train, y_train, model_type, cv=cv
#             )
#         return trained_models

#     def custom_cv_keras(self, X, y, cv=5):
#         kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
#         scores = []

#         # Convert to numpy arrays if they are pandas objects
#         if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
#             X = X.to_numpy()
#         if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
#             y = y.to_numpy()

#         for train, test in kfold.split(X, y):
#             # Create a new instance of KerasRegressor for each fold
#             model_clone = self.build_nn_model(X.shape[1])

#             # Fit on training fold
#             model_clone.fit(X[train], y[train])

#             # Evaluate on test fold
#             score = model_clone.score(X[test], y[test])
#             scores.append(-score)  # Negate the score to align with sklearn's convention

#         return np.array(scores)


# # Not using as not using nn anymore.
# def create_model(input_dim):
#     model = Sequential(
#         [
#             Input(shape=(input_dim,)),
#             Dense(64, activation="relu"),
#             Dropout(0.2),
#             Dense(32, activation="relu"),
#             Dropout(0.2),
#             Dense(16, activation="relu"),
#             Dense(1),
#         ]
#     )
#     model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
#     return model

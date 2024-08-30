from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from scikeras.wrappers import KerasRegressor
import os
import joblib


def create_model(input_dim):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


class ModelTrainer:
    def __init__(self):
        self.models = {
            "ridge": Ridge(),
            "rf": RandomForestRegressor(),
            "gb": GradientBoostingRegressor(),
            "nn": self.build_nn_model,
        }

    # Not working when trying to save the model to a pickle file, approaching it differently below.
    ## Worked, keeping newer version (defining the model creation function outside the class)

    def build_nn_model(self, input_dim):
        return create_model(input_dim)

    def save_keras_model(self, model, filename):
        print(f"Saving KerasRegressor model to {filename}")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Save the entire KerasRegressor object
        joblib.dump(model, filename)
        print(f"KerasRegressor model saved to {filename}")

    def train_model(self, X_train, y_train, model_type="rf", params=None):
        if model_type not in self.models:
            raise ValueError(f"Unsupported model type: {model_type}")

        if model_type == "nn":
            model = KerasRegressor(
                model=create_model,
                input_dim=X_train.shape[1],
                epochs=100,
                batch_size=32,
                verbose=0,
            )
        else:
            model = self.models[model_type]

        if params:
            model = GridSearchCV(model, params, cv=5, scoring="neg_mean_squared_error")

        model.fit(X_train, y_train)

        if isinstance(model, GridSearchCV):
            print(f"Best parameters for {model_type}: {model.best_params_}")
            return model.best_estimator_
        return model

    def train_multiple_models(self, X_train, y_train):
        trained_models = {}
        for model_type in self.models.keys():
            print(f"Training {model_type} model...")
            trained_models[model_type] = self.train_model(X_train, y_train, model_type)
        return trained_models

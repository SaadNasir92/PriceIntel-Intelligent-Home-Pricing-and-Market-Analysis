import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
)
from sklearn.compose import ColumnTransformer


class DataPreprocessor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = MinMaxScaler()

        # only using minmax now reduced model complexity, commenting out this.
        # self.scalers = {
        # "standard": StandardScaler(),
        # "robust": RobustScaler(),
        # "minmax": MinMaxScaler()
        # }
        self.categorical_encoder = None

    def load_data(self, filepath, sample_size=None):
        df = pd.read_csv(filepath)
        if sample_size and sample_size < len(df):
            return df.sample(n=sample_size, random_state=42)
        return df

    def load_and_sample_data(self, filepath, sample_size):
        # Read the entire dataset
        df = pd.read_csv(filepath)

        # Ensure 'Closing_Date' is in datetime format
        df["closing_date"] = pd.to_datetime(df["closing_date"])

        # Sort by closing date
        df = df.sort_values("closing_date")

        # Calculate the step size for sampling
        total_rows = len(df)
        step = max(1, total_rows // sample_size)

        # Sample the data
        sampled_indices = range(0, total_rows, step)
        df_sampled = df.iloc[sampled_indices].head(sample_size)

        return df_sampled

    def split_data(self, X, y, test_size=0.2):
        return train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

    # def handle_outliers(self, df):
    #     for column in df.select_dtypes(include=[np.number]):
    #         Q1 = df[column].quantile(0.25)
    #         Q3 = df[column].quantile(0.75)
    #         IQR = Q3 - Q1
    #         lower_bound = Q1 - 1.5 * IQR
    #         upper_bound = Q3 + 1.5 * IQR
    #         df[column] = df[column].clip(lower_bound, upper_bound)
    #     return df

    def scale_features(self, X_train, X_test):  # scaler_type ):
        # scaler = self.scalers[scaler_type]
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train), columns=X_train.columns
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=X_test.columns
        )
        return X_train_scaled, X_test_scaled

    # def impute_missing_values(self, X_train, X_test):
    #     X_train_imputed = self.imputer.fit_transform(X_train)
    #     X_test_imputed = self.imputer.transform(X_test)
    #     return X_train_imputed, X_test_imputed

    def preprocess_data(
        self,
        filepath,
        target_column,
        test_size=0.2,
        sample_size=None,
    ):
        df = self.load_data(filepath, sample_size)
        # df = self.load_and_sample_data(filepath, sample_size)

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        # Identify categorical columns
        categorical_columns = X.select_dtypes(include=["object", "category"]).columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns

        # # Split the data
        # X_train, X_test, y_train, y_test = self.split_data(X, y, test_size)

        # Split the data temporally
        split_index = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        # Create and fit the categorical encoder
        self.categorical_encoder = ColumnTransformer(
            [
                (
                    "onehot",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    categorical_columns,
                )
            ],
            remainder="passthrough",
        )

        # Encode categorical variables
        X_train_encoded = self.categorical_encoder.fit_transform(X_train)
        X_test_encoded = self.categorical_encoder.transform(X_test)

        # Get feature names after encoding
        onehot_columns = self.categorical_encoder.named_transformers_[
            "onehot"
        ].get_feature_names_out(categorical_columns)
        feature_names = np.concatenate([onehot_columns, numeric_columns])

        # Convert to DataFrame
        X_train_encoded = pd.DataFrame(X_train_encoded, columns=feature_names)
        X_test_encoded = pd.DataFrame(X_test_encoded, columns=feature_names)

        X_train_preprocessed, X_test_preprocessed = self.scale_features(
            X_train_encoded, X_test_encoded
        )

        # X_train_preprocessed, X_test_preprocessed = self.apply_preprocessing(
        # X_train_encoded, X_test_encoded, scaler_type
        # )
        # Convert all data to float32
        X_train_preprocessed = X_train_preprocessed.astype(np.float32)
        X_test_preprocessed = X_test_preprocessed.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        return X_train_preprocessed, X_test_preprocessed, y_train, y_test, feature_names

    # def apply_scaling(self, X_train, X_test, scaler_type="robust"):
    #     X_train_scaled, X_test_scaled, scaler = self.scale_features(
    #         X_train, X_test, scaler_type
    #     )
    #     return X_train_scaled, X_test_scaled, scaler

    # def apply_preprocessing(self, X_train, X_test, scaler_type):
    #     if scaler_type == "robust":
    #         # Skip outlier handling for Robust Scaler
    #         X_train_scaled, X_test_scaled = self.scale_features(
    #             X_train, X_test, scaler_type
    #         )
    #     else:
    #         # Handle outliers for other scalers
    #         X_train = self.handle_outliers(X_train)
    #         X_test = self.handle_outliers(X_test)
    #         X_train_scaled, X_test_scaled = self.scale_features(
    #             X_train, X_test, scaler_type
    #         )

    #     return X_train_scaled, X_test_scaled

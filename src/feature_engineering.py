from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import logging
import pandas as pd


class FeatureEngineer:
    def __init__(
        self,
        important_features=None,
        use_polynomial=True,
        polynomial_degree=2,
        k_best=50,
    ):
        self.important_features = important_features or [
            "square_footage",
            "upgrade_score",
            "annual_income",
            "credit_score",
        ]
        self.use_polynomial = use_polynomial
        self.polynomial_degree = polynomial_degree
        self.k_best = k_best
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.selected_feature_indices = None
        self.poly = None

    def create_interaction_terms(self, X, interaction_pairs):
        if X is None:
            return None
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X_with_interactions = X.copy()
        for f1, f2 in interaction_pairs:
            new_feature = X[:, f1] * X[:, f2]
            X_with_interactions = np.column_stack((X_with_interactions, new_feature))
        return X_with_interactions

    def create_polynomial_features(self, X):
        if not self.use_polynomial or X is None:
            return None
        if self.poly is None:
            self.poly = PolynomialFeatures(
                degree=self.polynomial_degree, include_bias=False, interaction_only=True
            )
            return self.poly.fit_transform(X)
        return self.poly.transform(X)

    def select_top_k_features(self, X, y):
        selector = SelectKBest(score_func=mutual_info_regression, k=self.k_best)
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
        self.logger.info(f"Selected {len(selected_features)} features")
        return X_selected, selected_features

    def engineer_features(self, X_train, X_test, y_train, feature_names):
        if X_test is None:
            raise ValueError("X_test cannot be None")

        # Convert to numpy arrays if they're pandas DataFrames
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()

        self.logger.info(f"Using features: {feature_names}")
        self.logger.info(
            f"X_train shape: {X_train.shape if X_train is not None else None}"
        )
        self.logger.info(f"X_test shape: {X_test.shape}")

        important_indices = [
            np.where(feature_names == f)[0][0]
            for f in self.important_features
            if f in feature_names
        ]

        # Create interaction terms only for important features
        interaction_pairs = [
            (i, j) for i in important_indices for j in important_indices if i < j
        ]
        X_train_interactions = self.create_interaction_terms(X_train, interaction_pairs)
        X_test_interactions = self.create_interaction_terms(X_test, interaction_pairs)

        self.logger.info(
            f"X_train shape after interactions: {X_train_interactions.shape}"
        )
        self.logger.info(
            f"X_test shape after interactions: {X_test_interactions.shape}"
        )

        # Create polynomial features only for important features
        X_train_important = X_train[:, important_indices]
        X_test_important = X_test[:, important_indices]

        if self.use_polynomial:
            X_train_poly = self.create_polynomial_features(X_train_important)
            X_test_poly = self.create_polynomial_features(X_test_important)

            self.logger.info(f"X_train shape after polynomial: {X_train_poly.shape}")
            self.logger.info(f"X_test shape after polynomial: {X_test_poly.shape}")

            # Combine original features with interaction terms and polynomial features
            X_train_combined = np.hstack([X_train, X_train_interactions, X_train_poly])
            X_test_combined = np.hstack([X_test, X_test_interactions, X_test_poly])
        else:
            # Combine original features with interaction terms only
            X_train_combined = np.hstack([X_train, X_train_interactions])
            X_test_combined = np.hstack([X_test, X_test_interactions])

        # Select top K features
        X_train_selected, self.selected_feature_indices = self.select_top_k_features(
            X_train_combined, y_train
        )
        X_test_selected = X_test_combined[:, self.selected_feature_indices]

        self.logger.info(f"Engineered features shape: {X_test_selected.shape[1]}")

        selected_feature_names = [
            feature_names[i]
            for i in self.selected_feature_indices
            if i < len(feature_names)
        ]

        self.logger.info(
            f"Selected features: {[feature_names[i] for i in self.selected_feature_indices if i < len(feature_names)]}"
        )

        return X_train_selected, X_test_selected, selected_feature_names


#  if self.use_polynomial:
#             X_train_poly = self.create_polynomial_features(X_train_important)
#             X_test_poly = self.create_polynomial_features(X_test_important)
#             X_train_combined = np.hstack([X_train, X_train_interactions, X_train_poly])
#             X_test_combined = np.hstack([X_test, X_test_interactions, X_test_poly])
#         else:
#             X_train_combined = np.hstack([X_train, X_train_interactions])
#             X_test_combined = np.hstack([X_test, X_test_interactions])

#         # Select top K features
#         X_train_selected, self.selected_feature_indices = self.select_top_k_features(X_train_combined, y_train)
#         X_test_selected = X_test_combined[:, self.selected_feature_indices]

#         self.logger.info(f"Engineered features shape: {X_test_selected.shape[1]}")
#         return X_train_selected, X_test_selected, [feature_names[i] for i in self.selected_feature_indices]

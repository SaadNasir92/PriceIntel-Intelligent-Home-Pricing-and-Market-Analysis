from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import logging
import pandas as pd

class FeatureEngineer:
    def __init__(self):
        self.required_features = ['square_footage', 'upgrade_score', 'annual_income', 'credit_score']
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

    def create_polynomial_features(self, X, degree=2):
        if X is None:
            return None
        if self.poly is None:
            self.poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            return self.poly.fit_transform(X)
        return self.poly.transform(X)

    def select_top_k_features(self, X, y, k=50):
        if X is None or y is None:
            return None, None
        
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_features = selector.get_support(indices=True)
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
        self.logger.info(f"X_train shape: {X_train.shape if X_train is not None else None}")
        self.logger.info(f"X_test shape: {X_test.shape}")

        # Find indices of the features we want to interact
        important_features = ['square_footage', 'upgrade_score', 'annual_income', 'credit_score']
        important_indices = [np.where(feature_names == f)[0][0] for f in important_features if f in feature_names]

        # Create interaction terms only for important features
        interaction_pairs = [(i, j) for i in important_indices for j in important_indices if i < j]
        X_train_interactions = self.create_interaction_terms(X_train, interaction_pairs) if X_train is not None else None
        X_test_interactions = self.create_interaction_terms(X_test, interaction_pairs)

        self.logger.info(f"X_train shape after interactions: {X_train_interactions.shape if X_train_interactions is not None else None}")
        self.logger.info(f"X_test shape after interactions: {X_test_interactions.shape}")

        # Create polynomial features only for important features
        X_train_important = X_train[:, important_indices] if X_train is not None else None
        X_test_important = X_test[:, important_indices]
        
        X_train_poly = self.create_polynomial_features(X_train_important)
        X_test_poly = self.create_polynomial_features(X_test_important)

        self.logger.info(f"X_train shape after polynomial: {X_train_poly.shape if X_train_poly is not None else None}")
        self.logger.info(f"X_test shape after polynomial: {X_test_poly.shape}")

        # Combine original features with interaction terms and polynomial features
        X_train_combined = np.hstack([X_train, X_train_interactions, X_train_poly]) if X_train is not None else None
        X_test_combined = np.hstack([X_test, X_test_interactions, X_test_poly])

        # Select top K features
        if X_train_combined is not None and y_train is not None:
            X_train_selected, self.selected_feature_indices = self.select_top_k_features(X_train_combined, y_train, k=50)
        else:
            X_train_selected = None

        X_test_selected = X_test_combined[:, self.selected_feature_indices] if self.selected_feature_indices is not None else X_test_combined

        self.logger.info(f"Engineered features shape: {X_test_selected.shape[1]}")
        
        return X_train_selected, X_test_selected, self.selected_feature_indices
from abc import ABC, abstractmethod
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd


class BaseClassifier(ABC):
    """
    Abstract base class for all trading classification models.
    This interface ensures all models have consistent methods for training,
    prediction, evaluation
    """

    def __init__(self, **kwargs):
        """
        Initialize the base classifier
        Args:
            **kwargs: Model-specific parameters
        """
        self.model = None
        self.is_trained = False
        self.feature_columns = None
        # Per-feature means computed on training data for NA imputation at inference
        self.feature_means: Optional[dict[str, float]] = None
        self.model_params = kwargs

    @abstractmethod
    def _create_model(self) -> Any:
        """
        Create the underlying ML model instance.
        Must be implemented by each specific classifier.
        Returns:
            Initialized model instance
        """
        pass

    @abstractmethod
    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to training data.
        Must be implemented by each specific classifier.
        Args:
            X: Feature matrix
            y: Target labels
        """
        pass

    @abstractmethod
    def _predict_model(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        Must be implemented by each specific classifier.
        Args:
            X: Feature matrix
        Returns:
            Predictions array
        """
        pass

    @abstractmethod
    def _predict_proba_model(self, x: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities from the trained model.
        Must be implemented by each specific classifier.
        Args:
            X: Feature matrix
        Returns:
            Probability matrix
        """
        pass

    def prepare_data(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training/prediction.
        Args:
            features_df: DataFrame from FeatureEngineer.create_features()
        Returns:
            X: Feature matrix
            y: Target labels (None if not available)
        """

        exclude_cols = ['Trade_ID', 'Profitable', 'EntryBar', 'ExitBar']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        X = features_df[feature_cols].copy()

        y: Optional[pd.Series] = None
        if 'Profitable' in features_df.columns:
            y = features_df['Profitable'].copy()

        if self.feature_columns is None:
            means = X.mean(numeric_only=True)
            X = X.fillna(means)
            self.feature_columns = feature_cols
            self.feature_means = means.to_dict()
        else:
            X = X.reindex(columns=self.feature_columns)
            if self.feature_means:
                X = X.fillna(pd.Series(self.feature_means))
            X = X.fillna(0)

        return X, y

    def train(self, features_df: pd.DataFrame, validation_split: float = 0.2) -> dict[str, Any]:
        """
        Train the classifier on the provided features.
        Args:
            features_df: DataFrame from FeatureEngineer.create_features()
            validation_split: Proportion of data for validation
        Returns:
            Dictionary with training results and metrics
        """
        if self.model is None:
            self.model = self._create_model()

        X, y = self.prepare_data(features_df)

        if y is None:
            raise ValueError("Training data must contain 'Profitable' column")

        # print(f"Training {self.__class__.__name__} with {len(X)} samples and {len(X.columns)} features")
        # print(f"Class distribution: {y.value_counts().to_dict()}")

        # Train the model
        self._fit_model(X.values, y.values)
        self.is_trained = True

        # Evaluate on training data
        train_predictions = self._predict_model(X.values)
        train_accuracy = np.mean(train_predictions == y.values)

        # print(f"Training accuracy: {train_accuracy:.4f}")

        results = {
            'model_name': self.__class__.__name__,
            'train_accuracy': train_accuracy,
            'n_samples': len(X),
            'n_features': len(X.columns),
            'feature_columns': self.feature_columns.copy(),
            'class_distribution': y.value_counts().to_dict()
        }

        return results

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict if trades are worth taking.
        Args:
            features_df: DataFrame with same structure as training data
        Returns:
            Predictions array (1 = worth taking, 0 = not worth taking)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X, _ = self.prepare_data(features_df)
        predictions = self._predict_model(X.values)
        return predictions

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        Args:
            features_df: DataFrame with same structure as training data
        Returns:
            Probability matrix
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X, _ = self.prepare_data(features_df)
        probabilities = self._predict_proba_model(X.values)
        return probabilities

    def evaluate(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """
        Evaluate the model on test data.
        Args:
            features_df: DataFrame with true labels
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        X, y_true = self.prepare_data(features_df)

        if y_true is None:
            raise ValueError("Evaluation data must contain 'Profitable' column")

        # Make predictions
        y_pred = self._predict_model(X.values)
        y_proba = self._predict_proba_model(X.values)

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_proba[:, 1]) if y_proba.shape[1] == 2 else None
        }

        return metrics

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'feature_means': self.feature_means,
            'model_params': self.model_params,
            'model_class': self.__class__.__name__
        }

        joblib.dump(model_data, filepath)
        # print(f"{self.__class__.__name__} saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)

        # Verify model class matches
        if model_data['model_class'] != self.__class__.__name__:
            raise ValueError(f"Model file contains {model_data['model_class']}, expected {self.__class__.__name__}")

        self.model = model_data['model']
        self.feature_columns = model_data.get('feature_columns')
        self.feature_means = model_data.get('feature_means')
        self.model_params = model_data.get('model_params', {})
        self.is_trained = True

        # print(f"{self.__class__.__name__} loaded from {filepath}")

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the model.
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.__class__.__name__,
            'is_trained': self.is_trained,
            'n_features': len(self.feature_columns) if self.feature_columns else None,
            'model_params': self.model_params
        }

    def __str__(self) -> str:
        """String representation of the classifier"""
        status = "trained" if self.is_trained else "untrained"
        n_features = len(self.feature_columns) if self.feature_columns else "unknown"
        return f"{self.__class__.__name__}(status={status}, features={n_features})"

    def __repr__(self) -> str:
        """Detailed representation of the classifier"""
        return self.__str__()

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseClassifier


class LogisticRegressionClassifier(BaseClassifier):
    """
    Binary classifier using sklearn LogisticRegression under the hood.
    """
    def _create_model(self) -> Any:
        # Reasonable defaults; allow overrides via self.model_params
        params = {
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": None,
        }
        params.update(self.model_params)
        return LogisticRegression(**params)

    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> None:
        if self.model is None:
            raise RuntimeError("Model instance is not created.")
        self.model.fit(x, y)
        self.is_trained = True

    def _predict_model(self, x: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        return self.model.predict(x)

    def _predict_proba_model(self, x: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before probability prediction.")
        return self.model.predict_proba(x)

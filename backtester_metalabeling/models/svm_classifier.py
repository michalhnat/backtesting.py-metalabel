
from typing import Any

import numpy as np
from sklearn.svm import SVC

from .base import BaseClassifier


class SVMClassifier(BaseClassifier):
    '''
    Support Vector Machine Classifier using sklearn's SVC.
    '''
    def _create_model(self) -> Any:
        params = {
            "probability": True,
            "class_weight": "balanced",
            "random_state": 42,
        }
        params.update(self.model_params)
        return SVC(**params)

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

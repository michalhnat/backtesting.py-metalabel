from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base import BaseClassifier


class RandomForrestModel(BaseClassifier):
    '''
    Random Forest Classifier using sklearn's RandomForestClassifier.
    '''
    def _create_model(self) -> Any:
        model = RandomForestClassifier(**self.model_params)

        if model is None:
            raise RuntimeError("Model instance is not created.")

        return model

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

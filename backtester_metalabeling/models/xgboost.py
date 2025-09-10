from typing import Any

import numpy as np
from xgboost import XGBClassifier

from .base import BaseClassifier


class XGBClassifierModel(BaseClassifier):
    '''
      XGBoost Classifier using xgboost's XGBClassifier.
    '''
    def _create_model(self) -> Any:
        return XGBClassifier(**self.model_params)

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

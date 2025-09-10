from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseClassifier


class EnsembleClassifier(BaseClassifier):
    """
    Ensemble over multiple BaseClassifier models.
    Strategies:
      - 'soft_voting': weighted average of probabilities
      - 'hard_voting': weighted majority vote
    """
    def __init__(
        self,
        base_models: list[tuple[str, BaseClassifier, float]],
        strategy: str = "soft_voting",
        auto_train_bases: bool = True,
    ):
        super().__init__()
        self.base_models = base_models
        self.strategy = strategy
        self.auto_train_bases = auto_train_bases

    def _fit_model(self, x: np.ndarray, y: np.ndarray) -> None:
        # Training handled in train(features_df); noop here.
        return

    def train(self, features_df: pd.DataFrame, validation_split: float = 0.0) -> dict[str, Any]:
        X, y = self.prepare_data(features_df)
        if y is None:
            raise ValueError("Training data must contain 'Profitable' column")

        if self.auto_train_bases:
            for _, model, _ in self.base_models:
                model.train(features_df, validation_split=0.0)

        self.is_trained = True
        acc = self._eval_voting_train_accuracy(features_df, y)

        return {
            "model_name": self.__class__.__name__,
            "strategy": self.strategy,
            "base_models": [name for name, _, _ in self.base_models],
            "train_accuracy": acc,
            "n_samples": len(X),
            "n_features": int(X.shape[1]),
        }

    def _eval_voting_train_accuracy(self, features_df: pd.DataFrame, y: pd.Series) -> float:
        preds = self.predict(features_df)
        return float(np.mean(preds == y.values))

    def _create_model(self):
        return None

    def _predict_model(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use predict(features_df: pd.DataFrame)")

    def _predict_proba_model(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use predict_proba(features_df: pd.DataFrame)")

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("EnsembleClassifier must be trained before predicting")

        if self.strategy == "hard_voting":
            return self._hard_vote(features_df)

        proba = self.predict_proba(features_df)[:, 1]
        return (proba >= 0.5).astype(int)

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("EnsembleClassifier must be trained before predicting")

        probs_list, weights = [], []
        for _, model, w in self.base_models:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(features_df)
                if p.ndim == 1 or p.shape[1] == 1:
                    p1 = p.ravel()
                    p = np.vstack([1 - p1, p1]).T
            else:
                pred = model.predict(features_df).astype(float)
                p = np.vstack([1 - pred, pred]).T
            probs_list.append(p)
            weights.append(float(w))

        weights = np.array(weights, dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        stacked = np.stack(probs_list, axis=0)  # (n_models, n_samples, 2)
        avg = np.tensordot(weights, stacked, axes=(0, 0))  # (n_samples, 2)
        return avg

    def _hard_vote(self, features_df: pd.DataFrame) -> np.ndarray:
        votes, weights = [], []
        for _, model, w in self.base_models:
            votes.append(model.predict(features_df).astype(int))
            weights.append(float(w))
        votes = np.stack(votes, axis=0)
        weights = np.array(weights)[:, None]
        weighted_sum = (votes * weights).sum(axis=0)
        thresh = weights.sum() / 2.0
        return (weighted_sum >= thresh).astype(int)


class EnsembleBuilder:
    """
    Builder for EnsembleClassifier.
    Example:
        ensemble = (
            EnsembleBuilder()
            .add_model("rf", rf_model, weight=2.0)
            .add_model("lr", lr_model, weight=1.0)
            .soft_voting()
            .build()
        )
        ensemble.train(features_df)
    """
    def __init__(self):
        self._base_models: list[tuple[str, BaseClassifier, float]] = []
        self._strategy: str = "soft_voting"
        self._auto_train_bases: bool = True

    def add_model(self, name: str, model: BaseClassifier | type[BaseClassifier], weight: float = 1.0) -> EnsembleBuilder:
        # Allow passing either an instance or a subclass; auto-instantiate subclasses
        if isinstance(model, type) and issubclass(model, BaseClassifier):
            model = model()  # type: ignore[call-arg]
        if not isinstance(model, BaseClassifier):
            raise TypeError("model must be a BaseClassifier instance or subclass")
        self._base_models.append((name, model, float(weight)))
        return self

    def soft_voting(self) -> EnsembleBuilder:
        self._strategy = "soft_voting"
        return self

    def hard_voting(self) -> EnsembleBuilder:
        self._strategy = "hard_voting"
        return self

    def auto_train_bases(self, flag: bool = True) -> EnsembleBuilder:
        self._auto_train_bases = bool(flag)
        return self

    def build(self) -> EnsembleClassifier:
        if not self._base_models:
            raise ValueError("No base models added to the ensemble")
        return EnsembleClassifier(
            base_models=self._base_models,
            strategy=self._strategy,
            auto_train_bases=self._auto_train_bases,
        )

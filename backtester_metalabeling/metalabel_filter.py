from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtester_metalabeling.models.base import BaseClassifier


def allow_trade(
    model: BaseClassifier,
    threshold: float = 0.5,
    features_fn: Optional[Callable[[object], Optional[pd.DataFrame]]] = None,
) -> Callable[[object], bool]:
    if model is None or not getattr(model, "is_trained", False):
        raise ValueError("Model must be provided and trained before using allow_trade.")
    if features_fn is None:
        raise ValueError("features_fn must be provided and return a 1-row DataFrame.")
    def gate(self_obj: object) -> bool:
        feats = features_fn(self_obj)
        if feats is None or not isinstance(feats, pd.DataFrame) or feats.empty:
            return True
        if len(feats) != 1:
            feats = feats.tail(1)
        if getattr(model, "feature_columns", None):
            for col in model.feature_columns:
                if col not in feats.columns:
                    feats[col] = np.nan
        feats = feats.fillna(0)
        proba = model.predict_proba(feats)
        p1 = float(proba[0, 1]) if proba.ndim == 2 and proba.shape[1] >= 2 else float(np.ravel(proba)[0])
        self_obj._ml_checks = int(getattr(self_obj, "_ml_checks", 0)) + 1
        if p1 >= threshold:
            self_obj._ml_pass = int(getattr(self_obj, "_ml_pass", 0)) + 1
        else:
            self_obj._ml_block = int(getattr(self_obj, "_ml_block", 0)) + 1
        try:
            self_obj._latest_ml_proba = p1
        except Exception:
            pass
        return p1 >= threshold
    return gate


def make_window_features_fn(
    data: pd.DataFrame,
    window_size: int,
) -> Callable[[object], Optional[pd.DataFrame]]:
    df_static = data
    def features_fn(self) -> Optional[pd.DataFrame]:
        df = getattr(getattr(self, "data", None), "df", None)
        if df is None:
            df = df_static
        i = int(getattr(self, "i", len(df) - 1))
        if i < window_size - 1:
            return None
        start = max(0, i - window_size + 1)
        window = df.iloc[start : i + 1]
        feat = {}
        for t, (_, row) in enumerate(window.iterrows()):
            for col in window.columns:
                feat[f"{col}_t{t}"] = row[col]
        num_cols = window.select_dtypes(include=[float, int]).columns
        for col in num_cols:
            s = window[col]
            feat[f"{col}_mean"] = s.mean()
            feat[f"{col}_std"] = s.std()
            feat[f"{col}_min"] = s.min()
            feat[f"{col}_max"] = s.max()
            if len(s) > 1:
                first = s.iloc[0]
                last = s.iloc[-1]
                feat[f"{col}_trend"] = (last - first) / first if first != 0 else 0
        feat["Trade_ID"] = i
        feat["EntryBar"] = i
        feat["ExitBar"] = None
        feat["Profitable"] = 0
        return pd.DataFrame([feat])
    return features_fn

import pandas as pd

from backtester_metalabeling.enhanced_strategy import EnhancedStrategy
from backtester_metalabeling.feature_engineering import FeatureEngineer
from backtester_metalabeling.metalabel_filter import allow_trade, make_window_features_fn
from backtester_metalabeling.models.base import BaseClassifier
from backtesting import Backtest, Strategy


class MetaLabeler:
    def __init__(self, model: BaseClassifier | None = None, strategy: type[Strategy] | None = None, window_size: int = 100, **kwargs):
        self.model = model
        self.strategy = strategy
        self.train_data = kwargs.pop('data', None)
        self.backtest_params = kwargs
        self.backtest_results = None
        self.__trades = None
        self.__features = None
        self.window_size = window_size
        self._model_trained = False
        self.last_train_stats: dict | None = None

    def set_model(self, model: BaseClassifier):
        self.model = model
        self._model_trained = False

    def set_strategy(self, strategy: type[Strategy]):
        self.strategy = strategy
        self._model_trained = False

    def set_train_data(self, data: pd.DataFrame):
        self.train_data = data
        self._model_trained = False

    def get_trades(self):
        return self.__trades

    def get_features(self):
        return self.__features

    def _backtest(self):
        if self.strategy is None:
            raise ValueError("Strategy is not set.")
        if self.train_data is None:
            raise ValueError("Training data must be provided.")
        params = dict(self.backtest_params)
        bt = Backtest(self.train_data, self.strategy, **params)
        self.backtest_results = bt.run()

    def _extract_trades(self):
        if self.backtest_results is None:
            raise RuntimeError("Backtest results not available.")
        self.__trades = self.backtest_results.get('_trades', [])

    def _feature_engineering(self):
        fe = FeatureEngineer(self.train_data, self.__trades, self.window_size)
        self.__features = fe.create_features()
        if not isinstance(self.__features, pd.DataFrame) or self.__features.empty:
            raise ValueError("No trades were generated on the training dataset for the selected strategy.")

    def _train_from_strategy(self, validation_split: float = 0.2):
        if self.model is None:
            raise ValueError("Model is not set.")
        self._backtest()
        self._extract_trades()
        self._feature_engineering()
        stats = self.model.train(self.__features, validation_split=validation_split)
        self.last_train_stats = stats
        self._model_trained = True
        return stats

    def create_filter(self, base_strategy: type[Strategy] | None, window_size: int = 100, threshold: float = 0.5, runtime_data=None):
        if base_strategy is None and self.strategy is None:
            raise ValueError("Base strategy must be provided.")
        if base_strategy is not None and base_strategy is not self.strategy:
            self.strategy = base_strategy
            self._model_trained = False
        retrain = (not self._model_trained) or (window_size != self.window_size)
        self.window_size = window_size
        if retrain:
            self._train_from_strategy()
        data_for_runtime = runtime_data if runtime_data is not None else self.train_data
        features_fn = make_window_features_fn(data_for_runtime, window_size)
        return allow_trade(self.model, threshold=threshold, features_fn=features_fn)

    def create_enhanced_strategy(self):
        return EnhancedStrategy



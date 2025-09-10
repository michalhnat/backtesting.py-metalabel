


class FeatureEngineer:
    def __init__(self, data, trades, window_size=3, indicators=None):
        self.data = data
        self.trades = trades
        self.window_size = window_size
        self.indicators = indicators if indicators is not None else []

    def create_features(self):
        import pandas as pd
        trades_df = self.trades
        if trades_df is None:
            trades_df = pd.DataFrame()
        if not hasattr(trades_df, 'iterrows'):
            try:
                trades_df = pd.DataFrame(list(trades_df) if trades_df is not None else [])
            except Exception:
                trades_df = pd.DataFrame()

        features = []
        self._attach_indicators()
        for index, trade in trades_df.iterrows():
            feature_dict = self._extract_features(trade)
            features.append(feature_dict)

        if features:
            return pd.DataFrame(features)
        return pd.DataFrame(columns=['Trade_ID', 'EntryBar', 'ExitBar', 'Profitable'])

    def _extract_features(self, trade):
        entry_bar = trade['EntryBar']
        profitable = 1 if trade['PnL'] > 0 else 0

        window_start = max(0, entry_bar - self.window_size)
        window_end = entry_bar

        data_window = self.data.iloc[window_start:window_end + 1].copy()

        feature_dict = {}

        feature_dict['Trade_ID'] = trade.name
        feature_dict['EntryBar'] = entry_bar
        feature_dict['ExitBar'] = trade['ExitBar'] if ('ExitBar' in trade) else None
        feature_dict['Profitable'] = profitable

        for i, (idx, row) in enumerate(data_window.iterrows()):
            for col in data_window.columns:
                feature_dict[f'{col}_t{i}'] = row[col]

        for col in data_window.select_dtypes(include=[float, int]).columns:
            feature_dict[f'{col}_mean'] = data_window[col].mean()
            feature_dict[f'{col}_std'] = data_window[col].std()
            feature_dict[f'{col}_min'] = data_window[col].min()
            feature_dict[f'{col}_max'] = data_window[col].max()
            if len(data_window) > 1:
                first = data_window[col].iloc[0]
                last = data_window[col].iloc[-1]
                feature_dict[f'{col}_trend'] = (last - first) / first if first != 0 else 0

        return feature_dict

    def _attach_indicators(self):
        for indicator in self.indicators:
            self.data[indicator.name] = indicator.calculate(self.data)

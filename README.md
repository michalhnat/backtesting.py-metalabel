# Backtesting.py - metalabeling

This fork intruduces simple implemention of [**meta-labeling**](https://en.wikipedia.org/wiki/Meta-Labeling) module build on top of [**Backtesting.py**](https://github.com/kernc/backtesting.py) library. 

The module integrates with Backtesting.py, allowing users to easily add meta-labeling to their existing strategies. It supports various machine learning models and provides utilities for feature engineering and signal filtering.


**Key features:**

- Easy integration with Backtesting.py strategies

- Support for multiple classifier models

- Ensemble model builder


## Usage

```sh

# clone this repo

cd backtesting.py-metalabel
uv sync

uv run python showcase.py
```

```py
#ASSET = GOOG

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 5)
        self.ma2 = self.I(SMA, price, 9)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


split_idx = int(len(ASSET) * 0.8)
train_data = ASSET.iloc[:split_idx].copy()
test_data = ASSET.iloc[split_idx:].copy()

rf_parms = {
    'n_estimators': 10000,
    'max_depth': 15,
    'random_state': 223145
}

xgb_params = {
    "n_estimators": 700,
    "learning_rate": 0.05,
}

ensemble = (
    EnsembleBuilder()
    .add_model("rf", RandomForrestModel(**rf_parms), weight=0.7)
    .add_model("xg", XGBClassifier(**xgb_params), weight=0.3)
    .soft_voting()
    .build()
)

model = ensemble 


curr_strat = SmaCross

ml = MetaLabeler(strategy=curr_strat, model=model, window_size=100,
                 data=train_data, commission=.002, exclusive_orders=True,
                 finalize_trades=True, cash=1_000_000)

gate = ml.create_filter(curr_strat, window_size=100, threshold=0.7, runtime_data=test_data)

enhanced_strategy = make_enhanced_strategy(curr_strat, gate)

bt = Backtest(test_data, enhanced_strategy, cash=1_000_000,
               commission=.002, exclusive_orders=True, finalize_trades=True)
```
Full script can be found in [showcase.py](./showcase.py).

## Results

Here is a markdown table comparing the results of the Base Strategy and Enhanced Strategy:

| Metric                   | Base Strategy         | Enhanced Strategy      |
|--------------------------|----------------------|-----------------------|
| Start                    | 2011-06-15 00:00:00  | 2011-06-15 00:00:00   |
| End                      | 2013-03-01 00:00:00  | 2013-03-01 00:00:00   |
| Duration                 | 625 days 00:00:00    | 625 days 00:00:00     |
| Exposure Time [%]        | 97.2093              | 97.2093               |
| Equity Final [$]         | 1,167,385.86         | <span style="color:green">1,631,710.29</span> |
| Equity Peak [$]          | 1,380,412.05         | 1,653,523.12          |
| Commissions [$]          | 217,971.24           | <span style="color:green">141,266.02</span> |
| Return [%]               | 16.74                | <span style="color:green">63.17</span>      |
| Buy & Hold Return [%]    | 66.98                | 66.98                 |
| Return (Ann.) [%]        | 9.49                 | 33.24                 |
| Volatility (Ann.) [%]    | 28.80                | 35.60                 |
| CAGR [%]                 | 6.44                 | 21.83                 |
| Sharpe Ratio             | 0.33                 | 0.93                  |
| Sortino Ratio            | 0.59                 | 1.96                  |
| Calmar Ratio             | 0.34                 | 1.81                  |
| Alpha [%]                | 15.82                | 21.85                 |
| Beta                     | 0.01                 | 0.62                  |
| Max. Drawdown [%]        | -27.54               | <span style="color:green">-18.41</span>     |
| Avg. Drawdown [%]        | -8.51                | -4.84                 |
| Max. Drawdown Duration   | 560 days 00:00:00    | 246 days 00:00:00     |
| Avg. Drawdown Duration   | 119 days 00:00:00    | 41 days 00:00:00      |
| # Trades                 | 46                   | <span style="color:green">27</span>           |
| Win Rate [%]             | 34.78                | <span style="color:green">55.56</span>      |
| Best Trade [%]           | 23.23                | 17.47                 |
| Worst Trade [%]          | -6.81                | -6.35                 |
| Avg. Trade [%]           | 0.34                 | 1.83                  |
| Max. Trade Duration      | 83 days 00:00:00     | 108 days 00:00:00     |
| Avg. Trade Duration      | 14 days 00:00:00     | 23 days 00:00:00      |
| Profit Factor            | 1.32                 | 3.16                  |
| Expectancy [%]           | 0.48                 | 1.97                  |
| SQN                      | 0.40                 | 1.90                  |
| Kelly Criterion          | 0.06                 | 0.37                  |
| _strategy                | SmaCross             | Enhanced_SmaCross     |
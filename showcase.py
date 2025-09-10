from backtester_metalabeling.enhanced_strategy import make_enhanced_strategy
from backtester_metalabeling.metalabeler import MetaLabeler
from backtester_metalabeling.models.classifiers import (
    EnsembleBuilder,
    RandomForrestModel,
    XGBClassifier,
)
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import GOOG, SMA

ASSET = GOOG

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
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 3,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "n_jobs": -1,
    "random_state": 42,
    "tree_method": "hist",
}

ensemble = (
    EnsembleBuilder()
    .add_model("rf", RandomForrestModel(**rf_parms), weight=0.7)
    .add_model("xg", XGBClassifier(**xgb_params), weight=0.3)
    .soft_voting()
    .build()
)


# ensemble = RandomForrestModel(**rf_parms)

curr_strat = SmaCross

ml = MetaLabeler(strategy=curr_strat, model=ensemble, window_size=100,
                 data=train_data, commission=.002, exclusive_orders=True,
                 finalize_trades=True, cash=1_000_000)

gate = ml.create_filter(curr_strat, window_size=100, threshold=0.7, runtime_data=test_data)

enhanced_strategy = make_enhanced_strategy(curr_strat, gate)

bt = Backtest(test_data, curr_strat, cash=1_000_000,
              commission=.002, exclusive_orders=True, finalize_trades=True)

btt = Backtest(test_data, enhanced_strategy, cash=1_000_000,
               commission=.002, exclusive_orders=True, finalize_trades=True)

res_ml = btt.run()
res_base = bt.run()

print(res_ml)
print("=" * 40)
print(res_base)

try:
    strat = res_ml.get('_strategy')
    print('ML checks/pass/block:', getattr(strat, '_ml_checks', 0), getattr(strat, '_ml_pass', 0), getattr(strat, '_ml_block', 0))
except Exception:
    pass


bt.plot()
btt.plot()

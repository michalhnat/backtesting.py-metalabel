from typing import Callable, Optional

from backtesting import Strategy

Gate = Callable[[Strategy], bool]


class MLEnhancedMixin(Strategy):
    ALLOW_ENTRY: Optional[Gate] = None

    _ml_checks: int = 0
    _ml_pass: int = 0
    _ml_block: int = 0

    def init(self):
        self._ml_checks = 0
        self._ml_pass = 0
        self._ml_block = 0
        return super().init()

    def _allow_entry(self) -> bool:
        gate = getattr(self.__class__, "ALLOW_ENTRY", None)
        if gate is None:
            return True
        self._ml_checks += 1
        try:
            ok = bool(gate(self))
        except Exception:
            ok = True
        if ok:
            self._ml_pass += 1
        else:
            self._ml_block += 1
        return ok

    def _is_entry_long(self) -> bool:
        pos = getattr(self, 'position', None)
        return (pos is None) or (not pos) or getattr(pos, 'is_short', False)

    def _is_entry_short(self) -> bool:
        pos = getattr(self, 'position', None)
        return (pos is None) or (not pos) or getattr(pos, 'is_long', False)

    def buy(self, *args, **kwargs):
        if self._is_entry_long() and not self._allow_entry():
            return None
        return super().buy(*args, **kwargs)

    def sell(self, *args, **kwargs):
        if self._is_entry_short() and not self._allow_entry():
            return None
        return super().sell(*args, **kwargs)

    def ml_stats(self) -> dict:
        return {
            "ml_checks": self._ml_checks,
            "ml_pass": self._ml_pass,
            "ml_block": self._ml_block,
            "ml_pass_rate": (self._ml_pass / self._ml_checks) if self._ml_checks else 0.0,
            "ml_block_rate": (self._ml_block / self._ml_checks) if self._ml_checks else 0.0,
        }


def make_enhanced_strategy(base_strategy: type[Strategy], ml_gate: Gate) -> type[Strategy]:
    name = f"Enhanced_{base_strategy.__name__}"
    return type(name, (MLEnhancedMixin, base_strategy), {"ALLOW_ENTRY": ml_gate})


EnhancedStrategy = make_enhanced_strategy

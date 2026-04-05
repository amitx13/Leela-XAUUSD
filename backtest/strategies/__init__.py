# This file is intentionally left as a redirect notice.
# The backtest/strategies/ subdirectory has been removed.
# All strategy logic lives in backtest/engine.py as standalone _evaluate_X() functions.
# This file will be removed in the next cleanup pass.
raise ImportError(
    "backtest.strategies subpackage has been removed. "
    "Import from backtest.strategies (the flat module) for ALL_STRATEGIES/STRATEGY_REGISTRY, "
    "or from backtest.engine for the actual evaluation functions."
)

"""
backtest — Phase 1B-1: Backtesting Framework for Leela XAUUSD.

Replay engine that simulates the entire trading system on historical data.
Reads stored OHLCV, computes indicators, runs regime engine, generates signals,
simulates execution, and produces per-strategy P&L curves.
"""
from backtest.engine import BacktestEngine
from backtest.results import BacktestResults
from backtest.monte_carlo import RiskOfRuinSimulator

__all__ = ["BacktestEngine", "BacktestResults", "RiskOfRuinSimulator"]

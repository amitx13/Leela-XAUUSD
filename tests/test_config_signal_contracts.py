"""Smoke tests: config keys used by signal/execution layers exist (no MT5/DB)."""
import pytest


def test_config_keys_used_by_signal_engine_exist():
    import config

    assert hasattr(config, "S1D_STOP_POINTS_MIN")
    assert hasattr(config, "S1F_STOP_POINTS")
    assert hasattr(config, "S7_MIN_RANGE_ATR_RATIO")
    assert hasattr(config, "S7_SIZE_MULTIPLIER")
    assert hasattr(config, "PREPLACEMENT_SPREAD_MULTIPLIER")


def test_build_initial_state_validates():
    from state import build_initial_state, validate_state_keys

    s = build_initial_state()
    validate_state_keys(s)


def test_position_manager_addon_strategies_not_blocked_by_trend_occupied():
    from engines.position_manager import can_open

    state = __import__("state", fromlist=["build_initial_state"]).build_initial_state()
    state["trend_family_occupied"] = True
    state["reversal_family_occupied"] = False
    ok, reason = can_open("S1D_PYRAMID", "LONG", 0.01, state)
    assert ok is True, reason

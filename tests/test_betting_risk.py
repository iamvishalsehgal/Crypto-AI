"""
Unit tests for BettingRiskManager.

Tests Kelly criterion, validation guards, halt mechanism, and state management.
Uses the standalone test pattern (no pytest).
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path for imports
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from dataclasses import dataclass
from typing import Any, Dict, List
from omnitrade.risk.betting_risk import BettingRiskManager
from omnitrade.config.settings import BettingSettings
from omnitrade.utils.odds import american_to_decimal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeBettingSettings:
    """Minimal stand-in for BettingSettings to avoid env-var dependencies."""
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.02
    min_edge_pct: float = 0.05
    max_daily_stakes: int = 20
    max_consecutive_losses: int = 8
    bankroll: float = 1000.0


@dataclass
class FakeSettings:
    """Minimal stand-in for Settings — only exposes the betting sub-object."""
    betting: Any = None


def make_manager(
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    max_stake_pct: float = 0.02,
    min_edge_pct: float = 0.05,
    max_daily_stakes: int = 20,
    max_consecutive_losses: int = 8,
) -> BettingRiskManager:
    bs = FakeBettingSettings(
        kelly_fraction=kelly_fraction,
        max_stake_pct=max_stake_pct,
        min_edge_pct=min_edge_pct,
        max_daily_stakes=max_daily_stakes,
        max_consecutive_losses=max_consecutive_losses,
        bankroll=bankroll,
    )
    settings = FakeSettings(betting=bs)
    return BettingRiskManager(settings)  # type: ignore[arg-type]


# ===================================================================
# american_to_decimal
# ===================================================================

def test_american_to_decimal_positive_odds() -> None:
    """+150 -> 2.5"""
    assert american_to_decimal(150) == 2.5


def test_american_to_decimal_negative_odds() -> None:
    """-200 -> 1.5"""
    assert american_to_decimal(-200) == 1.5


def test_american_to_decimal_zero_odds() -> None:
    """0 -> 2.0 (edge case)"""
    assert american_to_decimal(0) == 2.0


# ===================================================================
# Kelly criterion
# ===================================================================

def test_kelly_stake_known_inputs() -> None:
    """Probability=0.55, decimal_odds=2.0 -> edge=0.10, kelly=0.10 * bankroll."""
    rm = make_manager(bankroll=1000.0)
    stake = rm.kelly_stake(bankroll=1000.0, model_prob=0.55, odds_decimal=2.0)
    # f* = (1.0 * 0.55 - 0.45) / 1.0 = 0.10 -> stake = 1000 * 0.10 = 100.0
    assert abs(stake - 100.0) < 0.001, f"Expected 100.0, got {stake}"


def test_kelly_stake_no_edge() -> None:
    """Edge <= 0 => stake = 0."""
    rm = make_manager(bankroll=1000.0)
    stake = rm.kelly_stake(bankroll=1000.0, model_prob=0.40, odds_decimal=2.0)
    assert stake == 0.0, f"Expected 0.0, got {stake}"


def test_kelly_stake_negative_b_returns_zero() -> None:
    """Decimal odds <= 1.0 (b <= 0) => stake = 0."""
    rm = make_manager(bankroll=1000.0)
    stake = rm.kelly_stake(bankroll=1000.0, model_prob=0.99, odds_decimal=1.0)
    assert stake == 0.0, f"Expected 0.0, got {stake}"


def test_adjusted_stake_fractional_kelly() -> None:
    """Full Kelly * 0.25 fractional applied, then capped at max_stake_pct."""
    rm = make_manager(bankroll=1000.0, kelly_fraction=0.25, max_stake_pct=0.02)
    full = rm.kelly_stake(bankroll=1000.0, model_prob=0.55, odds_decimal=2.0)
    # full = 100.0, adjusted = 100.0 * 0.25 = 25.0, cap = 1000 * 0.02 = 20.0
    adjusted = rm.adjusted_stake(full, bankroll=1000.0)
    assert abs(adjusted - 20.0) < 0.001, f"Expected 20.0, got {adjusted}"


# ===================================================================
# validate_bet — guard conditions
# ===================================================================

def test_validate_bet_valid_passes() -> None:
    """Valid bet with good edge should be approved."""
    rm = make_manager(bankroll=1000.0)
    result = rm.validate_bet(
        symbol="TEST",
        model_prob=0.60,
        implied_prob=0.50,
        odds_american=-110,
        bankroll=1000.0,
        sport_key="soccer_epl",
    )
    assert result.approved, f"Expected approved, got reason: {result.reason}"
    assert result.adjusted_stake > 0
    assert result.risk_score == 0.0


def test_validate_bet_zero_bankroll() -> None:
    """Zero bankroll => rejected and halted."""
    rm = make_manager(bankroll=1000.0)
    result = rm.validate_bet(
        symbol="TEST",
        model_prob=0.60,
        implied_prob=0.50,
        odds_american=-110,
        bankroll=0.0,
    )
    assert not result.approved
    assert "Bankroll exhausted" in result.reason
    assert rm.is_halted


def test_validate_bet_negative_bankroll() -> None:
    """Negative bankroll => rejected and halted."""
    rm = make_manager(bankroll=1000.0)
    result = rm.validate_bet(
        symbol="TEST",
        model_prob=0.60,
        implied_prob=0.50,
        odds_american=-110,
        bankroll=-100.0,
    )
    assert not result.approved
    assert rm.is_halted


def test_validate_bet_daily_stake_limit() -> None:
    """Exceed max daily stakes => rejected."""
    rm = make_manager(bankroll=1000.0, max_daily_stakes=2)
    # Place two bets to fill the daily limit
    rm.record_bet("GAME1", 10.0)
    rm.record_bet("GAME2", 10.0)
    result = rm.validate_bet(
        symbol="GAME3",
        model_prob=0.60,
        implied_prob=0.50,
        odds_american=-110,
        bankroll=1000.0,
    )
    assert not result.approved
    assert "Max daily stakes" in result.reason
    assert result.risk_score == 0.9


def test_validate_bet_consecutive_loss_kill_switch() -> None:
    """Hit consecutive loss limit => halted and rejected."""
    rm = make_manager(bankroll=1000.0, max_consecutive_losses=3)
    for _ in range(3):
        rm.record_loss()
    result = rm.validate_bet(
        symbol="GAME4",
        model_prob=0.60,
        implied_prob=0.50,
        odds_american=-110,
        bankroll=1000.0,
    )
    assert not result.approved
    assert "consecutive losses" in result.reason
    assert result.risk_score == 1.0
    assert rm.is_halted


def test_validate_bet_edge_below_minimum() -> None:
    """Edge below min_edge_pct => rejected."""
    rm = make_manager(bankroll=1000.0, min_edge_pct=0.10)
    result = rm.validate_bet(
        symbol="TEST",
        model_prob=0.51,  # edge = 0.01
        implied_prob=0.50,
        odds_american=-110,
        bankroll=1000.0,
    )
    assert not result.approved
    assert "min" in result.reason
    assert result.risk_score == 0.5


def test_validate_bet_sport_concentration_limit() -> None:
    """Sport exposure > 40% of bankroll => rejected."""
    rm = make_manager(bankroll=100.0, kelly_fraction=1.0, max_stake_pct=0.5)
    # First bet for soccer_epl takes a large stake
    rm.record_bet("GAME1", stake=35.0, sport_key="soccer_epl")
    # Second bet for same sport should push over 40% cap
    result = rm.validate_bet(
        symbol="GAME2",
        model_prob=0.80,
        implied_prob=0.40,
        odds_american=-110,
        bankroll=100.0,
        sport_key="soccer_epl",
    )
    assert not result.approved
    assert "Sport exposure cap" in result.reason
    assert result.risk_score == 0.8


def test_validate_bet_kelly_zero_edge() -> None:
    """Stake <= 0 after Kelly => rejected."""
    rm = make_manager(bankroll=1000.0)
    # model_prob=0.50, implied_prob=0.50 => edge=0 < min_edge, but force past
    # the edge guard by setting min_edge_pct=0 and using a negative-edge bet
    # that still results in zero kelly
    rm._betting.min_edge_pct = 0.0  # type: ignore[attr-defined]
    result = rm.validate_bet(
        symbol="TEST",
        model_prob=0.51,
        implied_prob=0.50,
        odds_american=-110,
        bankroll=1000.0,
    )
    # model_prob=0.51, implied_prob=0.50 => edge=0.01 passes min_edge_pct=0,
    # but decimal 1.909 => b=0.909 => kelly negative => stake=0
    assert not result.approved
    assert "Kelly fraction" in result.reason
    assert result.risk_score == 0.3


def test_validate_bet_halted_rejects() -> None:
    """If manager is halted, validate_bet should short-circuit reject."""
    rm = make_manager(bankroll=1000.0)
    rm._halt("Test halt", 1000.0)
    result = rm.validate_bet(
        symbol="ANY",
        model_prob=0.99,
        implied_prob=0.01,
        odds_american=-110,
        bankroll=1000.0,
    )
    assert not result.approved
    assert "Halted" in result.reason
    assert result.risk_score == 1.0


# ===================================================================
# Halt mechanism
# ===================================================================

def test_is_halted_initial_false() -> None:
    """is_halted returns False on a fresh manager."""
    rm = make_manager()
    assert not rm.is_halted


def test_halt_sets_halted_and_reason() -> None:
    """_halt sets halted=True and stores the reason."""
    rm = make_manager()
    rm._halt("Bankroll exhausted", 0.0)
    assert rm.is_halted
    assert rm._halt_reason == "Bankroll exhausted"


def test_reset_daily_clears_halt() -> None:
    """reset_daily clears halt state and counters."""
    rm = make_manager(bankroll=500.0)
    rm._halt("Some reason", 100.0)
    rm.record_loss()
    rm.record_loss()
    rm.record_bet("G", 10.0)
    rm.reset_daily(bankroll=500.0)
    assert not rm.is_halted
    assert rm._halt_reason == ""
    assert rm._daily_stakes_count == 0
    assert rm._consecutive_losses == 0
    assert rm._daily_stake_total == 0.0


# ===================================================================
# State tracking
# ===================================================================

def test_get_status_returns_expected_keys() -> None:
    """get_status returns a dict with expected fields."""
    rm = make_manager(bankroll=1000.0)
    status = rm.get_status()
    assert "halted" in status
    assert "halt_reason" in status
    assert "consecutive_losses" in status
    assert "daily_stakes" in status
    assert "daily_stake_total" in status
    assert "max_consecutive_losses" in status
    assert "sport_exposure" in status
    assert "bankroll_history_len" in status
    assert not status["halted"]


def test_record_loss_increments_consecutive() -> None:
    """record_loss increments the consecutive loss counter."""
    rm = make_manager()
    assert rm.record_loss() == 1
    assert rm.record_loss() == 2


def test_record_loss_triggers_halt_at_limit() -> None:
    """When consecutive losses hit max, halt is triggered."""
    rm = make_manager(max_consecutive_losses=3)
    rm.record_loss()
    rm.record_loss()
    assert not rm.is_halted
    rm.record_loss()
    assert rm.is_halted
    assert "consecutive losses" in rm._halt_reason


def test_record_win_resets_consecutive() -> None:
    """record_win resets consecutive losses to 0."""
    rm = make_manager()
    rm.record_loss()
    rm.record_loss()
    rm.record_win()
    assert rm._consecutive_losses == 0


def test_record_bet_tracks_sport_exposure() -> None:
    """record_bet accumulates sport exposure."""
    rm = make_manager()
    rm.record_bet("G1", 10.0, "soccer_epl")
    rm.record_bet("G2", 15.0, "soccer_epl")
    assert rm._sport_exposure.get("soccer_epl") == 25.0
    status = rm.get_status()
    assert status["sport_exposure"]["soccer_epl"] == 25.0


def test_reset_daily_appends_bankroll_history() -> None:
    """reset_daily appends bankroll to history."""
    rm = make_manager(bankroll=1000.0)
    rm.reset_daily(bankroll=800.0)
    assert len(rm._bankroll_history) == 2  # initial + this call


# ===================================================================
# Edge rounding / special cases
# ===================================================================

def test_valid_bet_uses_correct_sport_key_no_exposure() -> None:
    """A valid bet with no prior sport exposure still passes."""
    rm = make_manager(bankroll=1000.0)
    result = rm.validate_bet(
        symbol="NEW",
        model_prob=0.65,
        implied_prob=0.45,
        odds_american=150,
        bankroll=1000.0,
        sport_key="new_sport",
    )
    # sport_key not in self._sport_exposure => no cap check triggered
    assert result.approved


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    tests = [
        test_american_to_decimal_positive_odds,
        test_american_to_decimal_negative_odds,
        test_american_to_decimal_zero_odds,
        test_kelly_stake_known_inputs,
        test_kelly_stake_no_edge,
        test_kelly_stake_negative_b_returns_zero,
        test_adjusted_stake_fractional_kelly,
        test_validate_bet_valid_passes,
        test_validate_bet_zero_bankroll,
        test_validate_bet_negative_bankroll,
        test_validate_bet_daily_stake_limit,
        test_validate_bet_consecutive_loss_kill_switch,
        test_validate_bet_edge_below_minimum,
        test_validate_bet_sport_concentration_limit,
        test_validate_bet_kelly_zero_edge,
        test_validate_bet_halted_rejects,
        test_is_halted_initial_false,
        test_halt_sets_halted_and_reason,
        test_reset_daily_clears_halt,
        test_get_status_returns_expected_keys,
        test_record_loss_increments_consecutive,
        test_record_loss_triggers_halt_at_limit,
        test_record_win_resets_consecutive,
        test_record_bet_tracks_sport_exposure,
        test_reset_daily_appends_bankroll_history,
        test_valid_bet_uses_correct_sport_key_no_exposure,
    ]

    passed = 0
    for t in tests:
        try:
            t()
            print(f"OK: {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"FAIL: {t.__name__} -- {exc}")
    print(f"\n{passed}/{len(tests)} tests passed")

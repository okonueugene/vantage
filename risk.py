"""
risk.py
───────
Risk Engine: three components.

1. DrawdownGovernor   — reduces Kelly when drawdown thresholds hit
2. MetaErrorPredictor — estimates P(model_error > theta) per bet
3. VolatilityRegime   — adjusts sizing by current variance environment
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Vantage.Risk")


# ──────────────────────────────────────────────
# 1. Drawdown Governor
# ──────────────────────────────────────────────
@dataclass
class DrawdownState:
    bankroll_history: List[float] = field(default_factory=list)
    peak_bankroll: float = 0.0
    current_drawdown_pct: float = 0.0
    kelly_multiplier: float = 1.0
    governor_active: bool = False
    governor_reason: str = ""


class DrawdownGovernor:
    """
    Reduces Kelly multiplier as drawdown deepens.
    Thresholds from Monte Carlo calibration:
      > 10% drawdown → 0.5x Kelly
      > 15% drawdown → 0.3x Kelly
      > 20% drawdown → 0.2x Kelly (near-stop)
    """
    THRESHOLDS = [
        (0.20, 0.20),   # (drawdown_pct, kelly_mult)
        (0.15, 0.30),
        (0.10, 0.50),
    ]

    def __init__(self, initial_bankroll: float):
        self.state = DrawdownState(
            bankroll_history=[initial_bankroll],
            peak_bankroll=initial_bankroll,
            kelly_multiplier=1.0,
        )

    def update(self, new_bankroll: float) -> DrawdownState:
        """Call after each settled bet. Updates state and returns current multiplier."""
        self.state.bankroll_history.append(new_bankroll)
        self.state.peak_bankroll = max(self.state.peak_bankroll, new_bankroll)

        dd = (self.state.peak_bankroll - new_bankroll) / self.state.peak_bankroll
        self.state.current_drawdown_pct = round(dd, 4)

        # Find applicable threshold
        new_mult = 1.0
        new_reason = ""
        for threshold, mult in sorted(self.THRESHOLDS, reverse=True):
            if dd >= threshold:
                new_mult = mult
                new_reason = f"Drawdown {dd*100:.1f}% ≥ {threshold*100:.0f}% → Kelly {mult}x"
                break

        if new_mult != self.state.kelly_multiplier:
            if new_mult < self.state.kelly_multiplier:
                logger.warning(f"Governor REDUCING Kelly: {new_reason}")
            else:
                logger.info(f"Governor EASING Kelly: drawdown recovered to {dd*100:.1f}%")

        self.state.kelly_multiplier = new_mult
        self.state.governor_active  = new_mult < 1.0
        self.state.governor_reason  = new_reason
        return self.state

    def get_multiplier(self) -> float:
        return self.state.kelly_multiplier

    def apply_to_stake(self, raw_stake: float) -> float:
        """Multiply raw stake by governor multiplier."""
        return round(raw_stake * self.state.kelly_multiplier, 2)


# ──────────────────────────────────────────────
# 2. Meta-Error Predictor
# ──────────────────────────────────────────────
@dataclass
class ErrorSignal:
    """Inputs to the meta-error model for one bet."""
    drs: float
    regime: str
    p_model_market_gap: float   # |p_model - p_market|
    motivation_delta: int
    is_compression_overs: bool
    league_sample_size: int     # how many results we have for this league


class MetaErrorPredictor:
    """
    Estimates P(model_error > theta) — uncertainty about our own prediction.
    Keeps it simple: L1-regularized logistic scoring (rule-based weights).
    
    This is NOT trying to predict outcomes. It's predicting when OUR MODEL
    is likely to be wrong, so we reduce Kelly accordingly.

    Regularization: weights are heavily shrunk toward 0 to prevent reflexivity.
    """

    # Regularized feature weights (L1-shrunk)
    # Higher score = higher model error risk = reduce Kelly
    WEIGHTS = {
        "drs":                    0.4,    # high disruption = model unreliable
        "p_gap":                  0.3,    # large model/market disagreement = risky
        "compression_overs":      0.8,    # historically worst error source
        "low_sample":             0.3,    # small league data = unreliable prior
        "high_motivation_away":   0.2,    # away team highly motivated = upset risk
    }

    ERROR_THRESHOLD = 0.35   # above this → activate Kelly reduction

    def score(self, signal: ErrorSignal) -> Tuple[float, str]:
        """
        Returns (error_probability_0_to_1, explanation).
        """
        score = 0.0
        factors = []

        # DRS contribution (continuous)
        drs_contrib = self.WEIGHTS["drs"] * signal.drs
        score += drs_contrib
        if signal.drs > 0.15:
            factors.append(f"High DRS ({signal.drs:.2f})")

        # Model/market gap
        gap_contrib = self.WEIGHTS["p_gap"] * signal.p_model_market_gap
        score += gap_contrib
        if signal.p_model_market_gap > 0.10:
            factors.append(f"Model↔Market gap {signal.p_model_market_gap:.2f}")

        # Compression overs — the core historical failure mode
        if signal.is_compression_overs:
            score += self.WEIGHTS["compression_overs"]
            factors.append("Compression overs (historically 80% miss rate)")

        # Small league sample — only when we have some data but it's thin.
        # When 0 results: no penalty (no calibration yet, not "unreliable league").
        if 1 <= signal.league_sample_size < 20:
            score += self.WEIGHTS["low_sample"]
            factors.append(f"Small sample ({signal.league_sample_size} results)")

        # Away motivation in high-deficit situations
        if signal.motivation_delta < -5:
            score += self.WEIGHTS["high_motivation_away"]
            factors.append(f"Away highly motivated (delta {signal.motivation_delta})")

        # Cap at 0.95
        score = min(score, 0.95)

        explanation = " | ".join(factors) if factors else "Normal confidence"
        return round(score, 3), explanation

    def kelly_reduction(self, error_prob: float) -> float:
        """
        Map error probability to Kelly multiplier.
        Continuous: reduction = 1 - error_prob (floored at 0.30).
        """
        reduction = max(1.0 - error_prob, 0.30)
        return round(reduction, 3)

    def apply(self, stake: float, signal: ErrorSignal) -> Tuple[float, float, str]:
        """
        Returns (adjusted_stake, error_prob, explanation).
        """
        error_prob, explanation = self.score(signal)
        if error_prob >= self.ERROR_THRESHOLD:
            mult    = self.kelly_reduction(error_prob)
            new_st  = round(stake * mult, 2)
            logger.info(f"Meta-error reducing stake {stake:.0f}→{new_st:.0f} "
                        f"(P_error={error_prob:.2f}: {explanation})")
            return new_st, error_prob, explanation
        return stake, error_prob, explanation


# ──────────────────────────────────────────────
# 3. Volatility Regime Sizer
# ──────────────────────────────────────────────
class VolatilityRegimeSizer:
    """
    Tracks rolling return volatility over last N bets.
    Scales Kelly down in high-vol regimes, up in stable ones.

    Complements the DrawdownGovernor — the governor reacts to losses,
    this reacts to variance (can trigger before losses materialise).
    """

    WINDOW = 30   # rolling bet window

    VOL_THRESHOLDS = [
        # (rolling_std_threshold, kelly_mult)
        (0.30, 0.70),   # very high vol → 70% Kelly
        (0.20, 0.85),   # elevated vol  → 85% Kelly
        (0.12, 1.00),   # normal        → 100%
        (0.00, 1.10),   # low vol       → 110% (slight upscale in stable runs)
    ]

    def __init__(self):
        self._returns: deque = deque(maxlen=self.WINDOW)
        self._vol_regime = "normal"

    def record_result(self, stake: float, pnl: float):
        """Record a bet result. pnl = profit/loss in bankroll % terms."""
        if stake > 0:
            self._returns.append(pnl / stake)

    def current_multiplier(self) -> Tuple[float, str]:
        """Returns (kelly_multiplier, regime_label)."""
        if len(self._returns) < 10:
            return 1.0, "insufficient_data"

        vol = self._rolling_std()
        for threshold, mult in sorted(self.VOL_THRESHOLDS, reverse=True):
            if vol >= threshold:
                regime = (
                    "high_volatility"    if vol >= 0.30
                    else "elevated"      if vol >= 0.20
                    else "normal"        if vol >= 0.12
                    else "low_volatility"
                )
                self._vol_regime = regime
                return mult, regime
        return 1.0, "normal"

    def _rolling_std(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        n    = len(self._returns)
        mean = sum(self._returns) / n
        variance = sum((r - mean) ** 2 for r in self._returns) / (n - 1)
        return round(math.sqrt(variance), 4)

    def apply_to_stake(self, raw_stake: float) -> Tuple[float, str]:
        mult, regime = self.current_multiplier()
        return round(raw_stake * mult, 2), regime


# ──────────────────────────────────────────────
# Unified Risk Engine (combines all three)
# ──────────────────────────────────────────────
class RiskEngine:
    """
    Single interface that applies all three risk layers in order:
      1. Volatility regime sizing
      2. Meta-error predictor
      3. Drawdown governor
    Layers multiply — if all three fire, the combined reduction can be significant.
    Floor: minimum 30% of original stake (prevents near-zero sizing).
    """

    MIN_STAKE_FLOOR = 0.30   # never go below 30% of raw stake

    def __init__(self, initial_bankroll: float):
        self.governor  = DrawdownGovernor(initial_bankroll)
        self.meta      = MetaErrorPredictor()
        self.vol_sizer = VolatilityRegimeSizer()

    def adjust_stake(
        self,
        raw_stake: float,
        error_signal: ErrorSignal,
    ) -> Dict:
        """
        Apply all three risk layers and return adjusted stake + audit trail.
        """
        original = raw_stake

        # Layer 1: Volatility regime
        stage1, vol_regime = self.vol_sizer.apply_to_stake(raw_stake)

        # Layer 2: Meta-error
        stage2, error_prob, error_explanation = self.meta.apply(stage1, error_signal)

        # Layer 3: Drawdown governor
        stage3 = self.governor.apply_to_stake(stage2)

        # Floor
        floor   = round(original * self.MIN_STAKE_FLOOR, 2)
        final   = max(stage3, floor)

        total_reduction = round((1 - final / original) * 100, 1) if original > 0 else 0

        return {
            "original_stake": original,
            "final_stake":    final,
            "total_reduction_pct": total_reduction,
            "vol_regime":     vol_regime,
            "vol_mult":       round(stage1 / original, 3) if original > 0 else 1,
            "error_prob":     error_prob,
            "error_explanation": error_explanation,
            "governor_mult":  self.governor.get_multiplier(),
            "governor_active": self.governor.state.governor_active,
            "current_drawdown_pct": self.governor.state.current_drawdown_pct,
        }

    def record_result(self, bankroll: float, stake: float, pnl: float):
        """Update all three risk layers after a result is known."""
        self.governor.update(bankroll)
        self.vol_sizer.record_result(stake, pnl)

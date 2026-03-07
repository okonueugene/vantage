"""
tracker.py
──────────
Post-bet tracking and edge attribution.

Tracks:
  1. CLV (Closing Line Value) — primary KPI
  2. Edge attribution per signal/regime/market
  3. Signal decay alerts (IR falling for a specific market)
  4. Rolling 200-bet IR
  5. Complexity score = active signals / IR

This is how you know if your edge is real and how long it lasts.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("Vantage.Tracker")

CLV_TARGET          = 0.52    # target: beat closing line >52% of bets
IR_TARGET           = 0.50    # target: IR > 0.5 sustained
EDGE_HALFLIFE_MIN   = 180     # warn if edge half-life < 180 bets
DECAY_WINDOW        = 50      # bets window for decay detection


@dataclass
class BetRecord:
    """Full audit record for a single settled bet."""
    bet_id:          str
    timestamp:       str
    home:            str
    away:            str
    league:          str
    market:          str
    bet_type:        str       # 'single' | 'parlay_leg'

    # Probabilities
    p_true:          float
    p_market:        float     # at bet placement
    p_closing:       float     # at market close (for CLV)

    # Odds
    placed_odds:     float
    closing_odds:    float
    decimal_odds:    float

    # Sizing
    stake:           float
    bankroll_at_bet: float

    # Adjustments applied
    regime:          str
    drs:             float
    ev_pct:          float
    kelly_mult:      float
    governor_mult:   float
    error_prob:      float

    # Result
    outcome:         int       # 1 = won, 0 = lost
    pnl:             float

    # Computed post-settlement
    clv:             float = 0.0   # closing_odds / placed_odds — 1 (positive = value)
    beat_close:      bool  = False


class EdgeTracker:

    def __init__(self, log_path: str = "bet_log.jsonl"):
        self.log_path = Path(log_path)
        self._records: List[BetRecord] = []
        self._load_existing()

    def _load_existing(self):
        """Load existing bet log on startup."""
        if self.log_path.exists():
            with open(self.log_path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        self._records.append(BetRecord(**d))
                    except Exception:
                        pass
            logger.info(f"Loaded {len(self._records)} historical records from {self.log_path}")

    def log_bet(self, record: BetRecord):
        """Log a settled bet and compute CLV."""
        # Compute CLV: positive = we got better price than closing
        if record.placed_odds > 0 and record.closing_odds > 0:
            record.clv = round(record.placed_odds / record.closing_odds - 1, 4)
            record.beat_close = record.clv > 0

        self._records.append(record)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record.__dict__) + "\n")

    # ──────────────────────────────────────────────
    # KPI Computations
    # ──────────────────────────────────────────────
    def clv_rate(self, last_n: int = 200) -> Dict:
        """% of bets that beat the closing line."""
        recent = [r for r in self._records[-last_n:] if r.closing_odds > 0]
        if not recent:
            return {"clv_rate": None, "n": 0, "mean_clv": None}

        beat_count = sum(1 for r in recent if r.beat_close)
        mean_clv   = sum(r.clv for r in recent) / len(recent)

        return {
            "clv_rate":  round(beat_count / len(recent), 4),
            "n":         len(recent),
            "mean_clv":  round(mean_clv, 4),
            "target":    CLV_TARGET,
            "on_track":  beat_count / len(recent) >= CLV_TARGET,
        }

    def information_ratio(self, last_n: int = 200) -> Dict:
        """Rolling IR = mean_return / std_return over last N bets."""
        recent = self._records[-last_n:]
        if len(recent) < 20:
            return {"ir": None, "n": len(recent)}

        returns = [r.pnl / r.stake for r in recent if r.stake > 0]
        if not returns:
            return {"ir": None, "n": 0}

        mean_r = sum(returns) / len(returns)
        std_r  = (sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5

        ir = round(mean_r / std_r, 3) if std_r > 0 else 0.0

        return {
            "ir":       ir,
            "n":        len(returns),
            "target":   IR_TARGET,
            "on_track": ir >= IR_TARGET,
        }

    def edge_attribution(self) -> pd.DataFrame:
        """
        Break down IR contribution by: market, regime, league, signal cluster.
        This is how you identify which signals are generating vs destroying edge.
        """
        if not self._records:
            return pd.DataFrame()

        rows = []
        for r in self._records:
            rows.append({
                "market":  r.market,
                "regime":  r.regime,
                "league":  r.league,
                "pnl":     r.pnl,
                "stake":   r.stake,
                "drs_bin": "high" if r.drs > 0.20 else "low",
                "outcome": r.outcome,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        def compute_ir(group):
            if len(group) < 5:
                return None
            returns = group["pnl"] / group["stake"].replace(0, float("nan"))
            mean_r  = returns.mean()
            std_r   = returns.std()
            return round(mean_r / std_r, 3) if std_r > 0 else 0.0

        # Attribution by market + regime
        attribution = (
            df.groupby(["market", "regime"])
            .apply(lambda g: pd.Series({
                "n_bets":   len(g),
                "hit_rate": round(g["outcome"].mean(), 3),
                "roi_pct":  round(g["pnl"].sum() / g["stake"].sum() * 100, 2) if g["stake"].sum() > 0 else 0,
                "ir":       compute_ir(g),
            }))
            .reset_index()
            .sort_values("roi_pct", ascending=False)
        )

        return attribution

    def signal_decay_alert(self) -> List[str]:
        """
        Detect if a market/regime combination is showing declining IR.
        Compare first half vs second half of last DECAY_WINDOW bets.
        """
        alerts = []
        recent = self._records[-DECAY_WINDOW:]
        if len(recent) < DECAY_WINDOW:
            return alerts

        half = DECAY_WINDOW // 2
        for market in set(r.market for r in recent):
            market_bets = [r for r in recent if r.market == market]
            if len(market_bets) < 10:
                continue

            first  = market_bets[:len(market_bets)//2]
            second = market_bets[len(market_bets)//2:]

            def roi(bets):
                total_stake = sum(b.stake for b in bets)
                return sum(b.pnl for b in bets) / total_stake if total_stake > 0 else 0

            roi_first  = roi(first)
            roi_second = roi(second)

            if roi_second < roi_first - 0.05:   # >5% ROI drop in second half
                alerts.append(
                    f"⚠️ Signal decay: {market} — ROI dropped from "
                    f"{roi_first*100:.1f}% to {roi_second*100:.1f}% "
                    f"(last {len(market_bets)} bets)"
                )

        return alerts

    def complexity_score(self, active_signals: int) -> Dict:
        """
        Complexity = active_signals / IR
        High complexity + flat IR → prune signals.
        """
        ir_data = self.information_ratio()
        ir      = ir_data.get("ir")

        if ir is None or ir <= 0:
            return {"complexity_score": None, "recommendation": "Need more data"}

        score = round(active_signals / ir, 2)
        rec   = "Prune signals" if score > 15 else "Acceptable complexity"

        return {
            "complexity_score": score,
            "active_signals":   active_signals,
            "ir":               ir,
            "recommendation":   rec,
        }

    def full_kpi_report(self, active_signals: int = 8) -> str:
        """Print full KPI dashboard."""
        clv   = self.clv_rate()
        ir    = self.information_ratio()
        decay = self.signal_decay_alert()
        comp  = self.complexity_score(active_signals)

        lines = [
            f"\n{'═'*55}",
            f"  KPI DASHBOARD — {len(self._records)} bets logged",
            f"{'═'*55}",
            f"  CLV Beat Rate:    {clv.get('clv_rate', 'N/A')} "
            f"(target {CLV_TARGET}) {'✅' if clv.get('on_track') else '❌'}",
            f"  Mean CLV:         {clv.get('mean_clv', 'N/A')}",
            f"  Rolling IR:       {ir.get('ir', 'N/A')} "
            f"(target {IR_TARGET}) {'✅' if ir.get('on_track') else '❌'}",
            f"  Complexity Score: {comp.get('complexity_score', 'N/A')} — {comp.get('recommendation')}",
            f"{'─'*55}",
        ]

        if decay:
            lines.append("  DECAY ALERTS:")
            for alert in decay:
                lines.append(f"    {alert}")
        else:
            lines.append("  No signal decay detected")

        lines.append(f"{'═'*55}\n")
        return "\n".join(lines)

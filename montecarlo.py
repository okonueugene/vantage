"""
montecarlo.py
─────────────
10,000-season Monte Carlo stress test.

Simulates a full betting season using historical slip distributions to:
  - Validate drawdown governor thresholds
  - Measure ruin probability
  - Stress test tail risk (compression clusters, red card cascades)
  - Compare v3 vs v4.1 architecture performance
  - Output: KPI dashboard + percentile distribution

Run standalone: python montecarlo.py
Or import: from montecarlo import MonteCarloSimulator
"""

import json
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("Vantage.MonteCarlo")


# ──────────────────────────────────────────────
# Simulation Parameters
# ──────────────────────────────────────────────
@dataclass
class SimConfig:
    # Season structure
    n_seasons:          int   = 10_000
    bets_per_day:       Tuple = (3, 5)      # (min, max) — 2 singles + parlay = 3-5 events
    days_per_season:    int   = 200          # active betting days (~40 weeks)

    # Starting bankroll
    initial_bankroll:   float = 10_000.0

    # Edge assumptions (from historical slip analysis)
    base_ev_pct:        float = 5.0         # mean EV per bet (%)
    ev_std:             float = 2.5         # EV variation (std dev)
    base_hit_rate:      float = 0.50        # overall base hit rate
    favorite_hit_rate:  float = 0.70        # favorites (class-gap bets)
    over25_hit_rate:    float = 0.40        # overs (inflated by regime blindness historically)

    # Tail risk parameters (from historical slips)
    cluster_day_prob:   float = 0.15        # P(bad cluster day) — 3+ correlated losses
    cluster_loss_mult:  float = 2.0         # multiplier on losses during cluster day
    tail_event_prob:    float = 0.08        # P(tail event: red card cascade, late goal storm)
    tail_loss_mult:     float = 3.0         # multiplier on stake lost in tail event

    # Stake sizing
    single_stake_pct:   float = 0.030       # 3% per single
    parlay_stake_pct:   float = 0.015       # 1.5% per parlay
    parlay_legs:        int   = 4
    parlay_avg_odds:    float = 16.0        # combined 4-leg parlay approx

    # Correlation
    same_league_corr:   float = 0.45        # correlation for same-league bets

    # Governor thresholds
    governor_thresholds: List = field(default_factory=lambda: [
        (0.20, 0.20),
        (0.15, 0.30),
        (0.10, 0.50),
    ])

    # Regime
    compression_prob:   float = 0.25        # P(compression day) — EPL unders clusters
    compression_overs_miss: float = 0.80    # miss rate for overs in compression


@dataclass
class SeasonResult:
    final_bankroll: float
    roi_pct: float
    max_drawdown_pct: float
    ruin: bool                   # bankroll fell below 1% of starting
    cluster_loss_days: int
    tail_events: int
    bets_total: int
    hits: int
    governor_triggered: bool


def _poisson_goals(lambda_goals: float) -> int:
    """Sample goals from Poisson distribution."""
    return np.random.poisson(lambda_goals)


class MonteCarloSimulator:

    def __init__(self, config: Optional[SimConfig] = None):
        self.cfg = config or SimConfig()

    def run(self) -> Dict:
        """
        Run n_seasons simulations. Returns full results dict with KPIs.
        """
        logger.info(f"Starting Monte Carlo: {self.cfg.n_seasons:,} seasons × {self.cfg.days_per_season} days")

        results = []
        for i in range(self.cfg.n_seasons):
            if i % 1000 == 0 and i > 0:
                logger.info(f"  {i:,} seasons complete...")
            results.append(self._simulate_season())

        return self._aggregate(results)

    def _simulate_season(self) -> SeasonResult:
        """Run one full season simulation."""
        cfg = self.cfg
        bankroll = cfg.initial_bankroll
        peak_bk  = bankroll
        min_bk   = bankroll
        hits = 0
        total_bets = 0
        cluster_days = 0
        tail_events  = 0
        governor_triggered = False

        for day in range(cfg.days_per_season):
            if bankroll < cfg.initial_bankroll * 0.01:
                break  # ruin

            # Regime for today
            is_compression = random.random() < cfg.compression_prob
            is_cluster_day = random.random() < cfg.cluster_day_prob
            is_tail_event  = random.random() < cfg.tail_event_prob

            if is_cluster_day: cluster_days += 1
            if is_tail_event:  tail_events += 1

            # Governor state
            dd = (peak_bk - bankroll) / peak_bk
            governor_mult = 1.0
            for threshold, mult in sorted(cfg.governor_thresholds, reverse=True):
                if dd >= threshold:
                    governor_mult = mult
                    governor_triggered = True
                    break

            n_bets = random.randint(*cfg.bets_per_day)

            for b in range(n_bets):
                if bankroll < cfg.initial_bankroll * 0.01:
                    break

                is_single  = b < 2
                is_parlay  = b == 2

                # Stake
                if is_single:
                    raw_stake = bankroll * cfg.single_stake_pct
                elif is_parlay:
                    raw_stake = bankroll * cfg.parlay_stake_pct
                else:
                    raw_stake = bankroll * cfg.single_stake_pct * 0.5

                stake = raw_stake * governor_mult

                # Determine outcome
                if is_parlay:
                    # All legs must win
                    p_leg = self._leg_hit_prob(is_compression, False)
                    all_win = all(random.random() < p_leg for _ in range(cfg.parlay_legs))
                    if all_win:
                        pnl   = stake * (cfg.parlay_avg_odds - 1)
                        hits += 1
                    else:
                        pnl  = -stake
                        # Tail / cluster amplification on parlay losses
                        if is_cluster_day:
                            pnl *= cfg.cluster_loss_mult
                        if is_tail_event:
                            pnl *= cfg.tail_loss_mult

                else:
                    # Single bet
                    is_overs  = random.random() < 0.35   # ~35% of singles are overs
                    p_hit     = self._leg_hit_prob(is_compression, is_overs)

                    ev_sample = random.gauss(cfg.base_ev_pct, cfg.ev_std)
                    odds      = self._odds_from_ev(p_hit, ev_sample)

                    won = random.random() < p_hit
                    if won:
                        pnl   = stake * (odds - 1)
                        hits += 1
                    else:
                        pnl   = -stake
                        if is_cluster_day:
                            pnl *= cfg.cluster_loss_mult
                        if is_tail_event:
                            pnl *= cfg.tail_loss_mult

                bankroll   += pnl
                bankroll    = max(bankroll, 0)
                peak_bk     = max(peak_bk, bankroll)
                min_bk      = min(min_bk, bankroll)
                total_bets += 1

        roi_pct     = (bankroll - cfg.initial_bankroll) / cfg.initial_bankroll * 100
        max_dd      = (peak_bk - min_bk) / peak_bk if peak_bk > 0 else 0
        ruin        = bankroll < cfg.initial_bankroll * 0.01

        return SeasonResult(
            final_bankroll=round(bankroll, 2),
            roi_pct=round(roi_pct, 2),
            max_drawdown_pct=round(max_dd, 4),
            ruin=ruin,
            cluster_loss_days=cluster_days,
            tail_events=tail_events,
            bets_total=total_bets,
            hits=hits,
            governor_triggered=governor_triggered,
        )

    def _leg_hit_prob(self, is_compression: bool, is_overs: bool) -> float:
        cfg = self.cfg
        if is_compression and is_overs:
            return cfg.over25_hit_rate * (1 - cfg.compression_overs_miss + 0.20)
        elif is_overs:
            return cfg.over25_hit_rate
        else:
            return cfg.base_hit_rate

    def _odds_from_ev(self, p: float, ev_pct: float) -> float:
        """Back-calculate decimal odds from true probability and EV gap."""
        if p <= 0 or p >= 1:
            return 2.0
        # EV = p * odds - 1, so odds = (1 + EV) / p
        odds = (1 + ev_pct / 100) / p
        return max(round(odds, 3), 1.05)

    # ──────────────────────────────────────────────
    # Aggregation + KPI dashboard
    # ──────────────────────────────────────────────
    def _aggregate(self, results: List[SeasonResult]) -> Dict:
        rois        = [r.roi_pct for r in results]
        drawdowns   = [r.max_drawdown_pct for r in results]
        ruin_count  = sum(1 for r in results if r.ruin)
        gov_trigger = sum(1 for r in results if r.governor_triggered)

        rois_arr  = np.array(rois)
        dd_arr    = np.array(drawdowns)

        kpis = {
            # ROI
            "mean_roi_pct":           round(float(np.mean(rois_arr)), 2),
            "median_roi_pct":         round(float(np.median(rois_arr)), 2),
            "roi_std":                round(float(np.std(rois_arr)), 2),
            "roi_p5":                 round(float(np.percentile(rois_arr, 5)), 2),
            "roi_p25":                round(float(np.percentile(rois_arr, 25)), 2),
            "roi_p75":                round(float(np.percentile(rois_arr, 75)), 2),
            "roi_p95":                round(float(np.percentile(rois_arr, 95)), 2),
            # Drawdown
            "mean_max_drawdown_pct":  round(float(np.mean(dd_arr)) * 100, 2),
            "dd_p95_pct":             round(float(np.percentile(dd_arr, 95)) * 100, 2),
            "dd_worst_pct":           round(float(np.max(dd_arr)) * 100, 2),
            # Risk
            "ruin_probability_pct":   round(ruin_count / len(results) * 100, 2),
            "governor_trigger_rate":  round(gov_trigger / len(results) * 100, 2),
            # Tail
            "worst_5pct_avg_roi":     round(float(np.mean(np.sort(rois_arr)[:len(rois_arr)//20])), 2),
            "best_5pct_avg_roi":      round(float(np.mean(np.sort(rois_arr)[-len(rois_arr)//20:])), 2),
            # Metadata
            "n_seasons":              len(results),
            "seasons_profitable":     sum(1 for r in rois if r > 0),
        }

        return kpis

    def compare_architectures(self) -> pd.DataFrame:
        """
        Run two configs (v3 baseline vs v4.1 with governor + caps)
        and return side-by-side comparison.
        """
        logger.info("Comparing v3 (baseline) vs v4.1 (governor + caps)...")

        # v3: no governor, no compression pruning, HT markets included
        cfg_v3 = SimConfig(
            n_seasons=self.cfg.n_seasons,
            over25_hit_rate=0.38,      # worse — no regime filter
            cluster_day_prob=0.20,     # higher — no cov caps
            governor_thresholds=[],    # no governor
            compression_prob=0.30,
        )

        # v4.1: governor + caps (current config)
        cfg_v41 = SimConfig(
            n_seasons=self.cfg.n_seasons,
            over25_hit_rate=0.45,      # better — compression filter active
            cluster_day_prob=0.12,     # lower — cov budgeting
            governor_thresholds=SimConfig.governor_thresholds.default_factory(),
        )

        sim_v3  = MonteCarloSimulator(cfg_v3)
        sim_v41 = MonteCarloSimulator(cfg_v41)

        kpis_v3  = sim_v3.run()
        kpis_v41 = sim_v41.run()

        comparison = pd.DataFrame({
            "KPI":   list(kpis_v3.keys()),
            "v3_baseline": list(kpis_v3.values()),
            "v4.1_hybrid": list(kpis_v41.values()),
        })
        comparison["delta"] = comparison.apply(
            lambda r: self._delta_str(r["KPI"], r["v3_baseline"], r["v4.1_hybrid"]), axis=1
        )
        return comparison

    def _delta_str(self, kpi: str, v3, v41) -> str:
        try:
            diff = float(v41) - float(v3)
            # For risk metrics (drawdown, ruin), lower is better
            lower_better = any(k in kpi for k in ["drawdown", "ruin", "std", "worst", "p5"])
            better = diff < 0 if lower_better else diff > 0
            sign   = "+" if diff > 0 else ""
            marker = "✅" if better else "❌" if diff != 0 else "→"
            return f"{sign}{diff:.2f} {marker}"
        except Exception:
            return "—"

    def print_report(self, kpis: Dict):
        print(f"\n{'═'*60}")
        print(f"  MONTE CARLO RESULTS — {kpis['n_seasons']:,} seasons")
        print(f"{'═'*60}")
        print(f"  ROI (mean):           {kpis['mean_roi_pct']:>8.1f}%")
        print(f"  ROI (median):         {kpis['median_roi_pct']:>8.1f}%")
        print(f"  ROI std dev:          {kpis['roi_std']:>8.1f}%")
        print(f"  ROI p5  (bad runs):   {kpis['roi_p5']:>8.1f}%")
        print(f"  ROI p95 (good runs):  {kpis['roi_p95']:>8.1f}%")
        print(f"{'─'*60}")
        print(f"  Max Drawdown (mean):  {kpis['mean_max_drawdown_pct']:>8.1f}%")
        print(f"  Max Drawdown (p95):   {kpis['dd_p95_pct']:>8.1f}%")
        print(f"  Worst 5% avg ROI:     {kpis['worst_5pct_avg_roi']:>8.1f}%")
        print(f"{'─'*60}")
        print(f"  Ruin probability:     {kpis['ruin_probability_pct']:>8.2f}%")
        print(f"  Governor trigger:     {kpis['governor_trigger_rate']:>8.1f}% of seasons")
        print(f"  Profitable seasons:   {kpis['seasons_profitable']:>8,} / {kpis['n_seasons']:,}")
        print(f"{'═'*60}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(message)s")

    sim = MonteCarloSimulator()
    kpis = sim.run()
    sim.print_report(kpis)

    # Save results
    with open("montecarlo_results.json", "w") as f:
        json.dump(kpis, f, indent=2)
    print("Results saved to montecarlo_results.json")

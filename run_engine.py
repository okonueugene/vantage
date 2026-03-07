"""
run_engine.py
─────────────
Vantage Engine v4.1 — Full Production Orchestrator

Full pipeline:
  1. Fetch structured fixtures (ESPN API + TheSportsDB + API-Football)
  2. Normalize + odds enrich
  3. Validate + confidence score
  4. Predict p_true for all markets (Bayesian ensemble)
  5. Apply risk adjustments (meta-error, volatility, governor)
  6. Build dual output (2 singles + 4-leg parlay)
  7. Print output slip + log for CLV tracking

Usage:
  python run_engine.py                    # live run (today's matches)
  python run_engine.py --test             # test mode (synthetic data)
  python run_engine.py --montecarlo       # 10,000-season stress test
  python run_engine.py --kpi              # print KPI dashboard
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

# ── Internal modules ────────────────────────────────────────────────────────────
from structured_fetcher import StructuredFetcher
from odds_fetcher import OddsFetcher
from normalizer import normalize_dataframe
from validator import Validator
from scrape_fallback import ScrapeFallback
from predictor import Predictor
from portfolio import PortfolioBuilder, Portfolio, SingleBet, ParlayBet
from risk import RiskEngine, ErrorSignal
from tracker import EdgeTracker, BetRecord
from montecarlo import MonteCarloSimulator, SimConfig

# ──────────────────────────────────────────────
# Config — set your keys and bankroll here
# ──────────────────────────────────────────────
API_FOOTBALL_KEY = ""      # api-sports.io free key
THE_ODDS_API_KEY = ""      # the-odds-api.com free key
BANKROLL_KES     = 10_000  # current bankroll in KES

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)-22s | %(message)s",
    handlers=[
        logging.FileHandler("vantage.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("Vantage.Main")


def run_live(bankroll: float = BANKROLL_KES):
    """Full live pipeline."""
    scan_id  = f"V41_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"\n{'='*65}")
    logger.info(f"  VANTAGE ENGINE v4.1 | {scan_id}")
    logger.info(f"  Bankroll: KES {bankroll:,.0f}")
    logger.info(f"{'='*65}")

    # ── Layer 1: Fetch ──────────────────────────────────────────────────────────
    fetcher = StructuredFetcher(api_football_key=API_FOOTBALL_KEY)
    raw_df  = fetcher.fetch_all()

    # ── Normalize ───────────────────────────────────────────────────────────────
    norm_df = normalize_dataframe(raw_df) if not raw_df.empty else raw_df

    # ── Odds ────────────────────────────────────────────────────────────────────
    odds_f  = OddsFetcher(
        api_football_key=API_FOOTBALL_KEY,
        the_odds_key=THE_ODDS_API_KEY,
    )
    enriched_df = odds_f.enrich_with_odds(norm_df) if not norm_df.empty else norm_df

    # ── Fallback ────────────────────────────────────────────────────────────────
    fallback = ScrapeFallback()
    fallback_df = fallback.fetch(min_structured_count=15, structured_df=enriched_df)
    if not fallback_df.empty:
        fallback_norm = normalize_dataframe(fallback_df)
        enriched_df   = pd.concat([enriched_df, fallback_norm], ignore_index=True)

    # ── Validate ────────────────────────────────────────────────────────────────
    validator = Validator(require_odds=False, min_confidence="medium")
    clean_df  = validator.validate(enriched_df)
    if clean_df.empty:
        logger.critical("No valid fixtures — aborting")
        return

    # Stub regime/DRS onto clean_df if not present (real values come from enrichment)
    if "regime" not in clean_df.columns:
        clean_df["regime"] = "neutral"
    if "drs" not in clean_df.columns:
        clean_df["drs"] = 0.12
    if "motivation_delta" not in clean_df.columns:
        clean_df["motivation_delta"] = 0

    # ── Predict ─────────────────────────────────────────────────────────────────
    predictor = Predictor()
    pred_df   = predictor.predict_all(clean_df)

    if pred_df.empty:
        logger.warning("No bets with edge found today — no output generated")
        _print_no_output(clean_df)
        return

    # ── Risk Engine ─────────────────────────────────────────────────────────────
    risk = RiskEngine(initial_bankroll=bankroll)

    # Adjust each candidate's stake through risk engine
    adjusted_rows = []
    for _, row in pred_df.iterrows():
        signal = ErrorSignal(
            drs=float(row.get("drs", 0.12)),
            regime=row.get("regime", "neutral"),
            p_model_market_gap=abs(float(row.get("p_model", 0.5)) - float(row.get("p_market", 0.5))),
            motivation_delta=int(row.get("motivation_delta", 0)),
            is_compression_overs=(row.get("regime") == "compression" and row.get("market") == "over25"),
            league_sample_size=50,   # TODO: wire to actual league history count
        )
        # Compute raw stake
        p    = float(row["p_true"])
        odds = float(row["decimal_odds"])
        q    = 1 - p
        raw_kelly = max((p * odds - 1) / (odds - 1), 0) if odds > 1 else 0
        raw_stake = min(raw_kelly * 0.5, 0.03) * bankroll

        adjustment = risk.adjust_stake(raw_stake, signal)
        row_dict = row.to_dict()
        row_dict["risk_adjusted_stake"] = adjustment["final_stake"]
        row_dict["risk_reduction_pct"]  = adjustment["total_reduction_pct"]
        row_dict["error_prob"]          = adjustment["error_prob"]
        adjusted_rows.append(row_dict)

    pred_df = pd.DataFrame(adjusted_rows)

    # ── Portfolio ────────────────────────────────────────────────────────────────
    # Use risk-adjusted stakes in portfolio builder
    portfolio_builder = PortfolioBuilder(bankroll=bankroll)
    portfolio = portfolio_builder.build(pred_df)

    # ── Print Output Slip ───────────────────────────────────────────────────────
    _print_slip(portfolio, bankroll, scan_id)

    # ── Save ────────────────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    pred_df.to_csv(f"predictions_{ts}.csv", index=False)
    logger.info(f"Predictions saved to predictions_{ts}.csv")

    return portfolio


def _print_slip(portfolio: Portfolio, bankroll: float, scan_id: str):
    """Print the final betting slip."""
    print(f"\n{'═'*65}")
    print(f"  🎯 VANTAGE OUTPUT SLIP | {scan_id}")
    print(f"  Bankroll: KES {bankroll:,.0f}")
    print(f"{'═'*65}")

    # Singles
    print(f"\n  📌 SINGLES ({len(portfolio.singles)})\n")
    total_singles = 0
    for i, s in enumerate(portfolio.singles, 1):
        print(f"  [{i}] {s.home} vs {s.away}")
        print(f"      Market:  {s.market.replace('_', ' ').title()}")
        print(f"      League:  {s.league}")
        print(f"      Odds:    {s.decimal_odds} | p_true: {s.p_true:.1%}")
        print(f"      EV:      +{s.ev_pct:.1f}% | DRS: {s.drs:.2f} | Regime: {s.regime}")
        print(f"      Stake:   KES {s.stake_kes:,.0f} ({s.stake_units*100:.1f}% bankroll)")
        print(f"      Return:  KES {s.stake_kes * s.decimal_odds:,.0f} if wins")
        print(f"      Note:    {s.rationale}")
        print()
        total_singles += s.stake_kes

    # Parlay
    if portfolio.parlay:
        p = portfolio.parlay
        print(f"  🎰 4-LEG PARLAY\n")
        for i, leg in enumerate(p.legs, 1):
            print(f"    Leg {i}: {leg.home} vs {leg.away}")
            print(f"            {leg.market.replace('_',' ').title()} @ {leg.decimal_odds} | EV {leg.ev_pct:.1f}%")
        print(f"\n    Combined odds:   {p.combined_odds:.1f}x")
        print(f"    Combined p_true: {p.combined_p_true:.4f} ({p.combined_p_true*100:.2f}%)")
        print(f"    Parlay EV:       {p.combined_ev_pct:+.1f}%")
        print(f"    Stake:           KES {p.stake_kes:,.0f} ({p.stake_units*100:.1f}% bankroll)")
        print(f"    Potential win:   KES {p.stake_kes * p.combined_odds:,.0f}")
        if p.compression_flag:
            print(f"    ⚠️  Compression flag — stake reduced to {p.size_reduction*100:.0f}%")
    else:
        print(f"  ❌ Parlay not built")
        if portfolio.parlay:
            print(f"     Reason: {portfolio.parlay.veto_reason}")

    # Summary
    print(f"\n{'─'*65}")
    print(f"  Total exposure: KES {portfolio.total_exposure_kes:,.0f} "
          f"({portfolio.total_exposure_pct*100:.1f}% bankroll)")

    # Regime exposure
    if portfolio.regime_exposure:
        regime_str = " | ".join(f"{k}: {v*100:.0f}%" for k, v in portfolio.regime_exposure.items())
        print(f"  Regime mix:     {regime_str}")

    print(f"{'═'*65}\n")


def _print_no_output(clean_df: pd.DataFrame):
    """Print diagnostic when no edge found."""
    print(f"\n{'═'*65}")
    print(f"  ⚪ NO OUTPUT — No bets with edge found today")
    print(f"{'═'*65}")
    print(f"  Fixtures scanned: {len(clean_df)}")
    print(f"  Leagues: {', '.join(clean_df['league'].unique()[:5])}")
    print(f"  Tip: EV threshold is {3.0}%. Raise odds range or lower threshold.")
    print(f"{'═'*65}\n")


def run_test():
    """Test mode with synthetic fixture data to verify pipeline end-to-end."""
    logger.info("TEST MODE — synthetic fixtures")

    # Create synthetic fixtures covering key edge cases
    test_fixtures = pd.DataFrame([
        {"home": "Manchester City", "away": "Arsenal",        "league": "Premier League",
         "odds_1": 1.80, "odds_X": 3.60, "odds_2": 4.50,
         "odds_over25": 1.72, "odds_u25": 2.15,
         "regime": "neutral", "drs": 0.10, "motivation_delta": 2,
         "has_odds": True, "confidence": "high", "source": "test",
         "canonical_id": "MANU_ARS_TEST"},

        {"home": "Real Madrid",     "away": "Barcelona",      "league": "La Liga",
         "odds_1": 2.10, "odds_X": 3.40, "odds_2": 3.30,
         "odds_over25": 1.68, "odds_u25": 2.25,
         "regime": "expansion", "drs": 0.14, "motivation_delta": 5,
         "has_odds": True, "confidence": "high", "source": "test",
         "canonical_id": "REAL_BAR_TEST"},

        {"home": "Bayern Munich",   "away": "Dortmund",       "league": "Bundesliga",
         "odds_1": 1.65, "odds_X": 3.90, "odds_2": 5.20,
         "odds_over25": 1.60, "odds_u25": 2.40,
         "regime": "expansion", "drs": 0.08, "motivation_delta": 3,
         "has_odds": True, "confidence": "high", "source": "test",
         "canonical_id": "BAY_DOR_TEST"},

        {"home": "Inter Milan",     "away": "Juventus",       "league": "Serie A",
         "odds_1": 1.95, "odds_X": 3.50, "odds_2": 3.80,
         "odds_over25": 1.95, "odds_u25": 1.92,
         "regime": "compression", "drs": 0.19, "motivation_delta": -2,
         "has_odds": True, "confidence": "high", "source": "test",
         "canonical_id": "INT_JUV_TEST"},

        {"home": "PSG",             "away": "Marseille",      "league": "Ligue 1",
         "odds_1": 1.55, "odds_X": 4.20, "odds_2": 5.50,
         "odds_over25": 1.75, "odds_u25": 2.10,
         "regime": "neutral", "drs": 0.22, "motivation_delta": 8,
         "has_odds": True, "confidence": "high", "source": "test",
         "canonical_id": "PSG_MAR_TEST"},

        # Edge case: compression + overs (should be filtered)
        {"home": "Atletico Madrid", "away": "Villarreal",     "league": "La Liga",
         "odds_1": 1.88, "odds_X": 3.60, "odds_2": 4.10,
         "odds_over25": 2.05, "odds_u25": 1.82,
         "regime": "compression", "drs": 0.15, "motivation_delta": 1,
         "has_odds": True, "confidence": "medium", "source": "test",
         "canonical_id": "ATL_VIL_TEST"},
    ])

    predictor = Predictor()
    pred_df   = predictor.predict_all(test_fixtures)

    print(f"\nPredictions generated: {len(pred_df)} with edge")
    if not pred_df.empty:
        display = pred_df[["home", "away", "league", "market", "p_true", "p_market",
                           "ev_pct", "regime", "drs"]].copy()
        print(display.to_string(index=False))

    risk      = RiskEngine(initial_bankroll=BANKROLL_KES)
    portfolio = PortfolioBuilder(bankroll=BANKROLL_KES).build(pred_df)
    _print_slip(portfolio, BANKROLL_KES, "TEST_RUN")


def run_montecarlo():
    sim  = MonteCarloSimulator(SimConfig(n_seasons=10_000))
    kpis = sim.run()
    sim.print_report(kpis)


def run_kpi():
    tracker = EdgeTracker()
    print(tracker.full_kpi_report(active_signals=8))
    attribution = tracker.edge_attribution()
    if not attribution.empty:
        print("\nEdge Attribution by Market + Regime:")
        print(attribution.to_string(index=False))


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vantage Engine v4.1")
    parser.add_argument("--test",        action="store_true", help="Run with synthetic test data")
    parser.add_argument("--montecarlo",  action="store_true", help="Run 10,000-season stress test")
    parser.add_argument("--kpi",         action="store_true", help="Print KPI dashboard from bet log")
    parser.add_argument("--bankroll",    type=float, default=BANKROLL_KES, help="Bankroll in KES")
    args = parser.parse_args()

    if args.montecarlo:
        run_montecarlo()
    elif args.test:
        run_test()
    elif args.kpi:
        run_kpi()
    else:
        run_live(bankroll=args.bankroll)

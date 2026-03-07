"""
engine.py
─────────
Vantage Parlay Engine v4.1 — Hybrid Architecture Orchestrator

Data flow:
  ┌─────────────────────────────────────────────────────────┐
  │  Layer 1: Structured APIs (structured_fetcher.py)       │
  │    ESPN API → TheSportsDB → API-Football                 │
  └──────────────────────┬──────────────────────────────────┘
                         │ JSON fixtures
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Normalize (normalizer.py)                              │
  │    Canonical team names, kickoff format, canonical_id   │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Odds Enrichment (odds_fetcher.py)                      │
  │    The Odds API or API-Football odds endpoint           │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Validate (validator.py)                                │
  │    Junk filter, time window, dedup, confidence score    │
  └──────────────────────┬──────────────────────────────────┘
                         │ < 20 matches?
                         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Fallback (scrape_fallback.py) — only if needed         │
  │    Soccerway + LiveScore light scrape                   │
  └──────────────────────┬──────────────────────────────────┘
                         │
                         ▼
                  Clean fixture CSV/JSON
                  (ready for Filter → Predictor)
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Module imports ──────────────────────────────────────────────────────────────
from structured_fetcher import StructuredFetcher
from odds_fetcher import OddsFetcher
from normalizer import normalize_dataframe
from validator import Validator
from scrape_fallback import ScrapeFallback

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler("vantage_engine.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("VantageV5.Engine")


class VantageEngine:

    # ── Config ─────────────────────────────────────────────────────────────────
    MIN_STRUCTURED_MATCHES = 20   # Below this → activate fallback scraper
    REQUIRE_ODDS           = False # Set True to drop matches with no odds
    MIN_CONFIDENCE         = "medium"  # 'low' | 'medium' | 'high'

    # API keys — set here or use environment variables
    API_FOOTBALL_KEY = ""  # os.getenv("API_FOOTBALL_KEY", "")
    THE_ODDS_API_KEY = ""  # os.getenv("THE_ODDS_API_KEY", "")

    def __init__(self):
        self.scan_time = datetime.now(timezone.utc)
        self.scan_id   = f"V41_{self.scan_time.strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(".")

        # Module instances
        self.fetcher  = StructuredFetcher(api_football_key=self.API_FOOTBALL_KEY)
        self.odds     = OddsFetcher(
            api_football_key=self.API_FOOTBALL_KEY,
            the_odds_key=self.THE_ODDS_API_KEY,
        )
        self.validator = Validator(
            require_odds=self.REQUIRE_ODDS,
            min_confidence=self.MIN_CONFIDENCE,
        )
        self.fallback  = ScrapeFallback()

    def run(self) -> pd.DataFrame:
        """
        Full pipeline execution.
        Returns clean, validated fixture DataFrame ready for Layer 2 (Filter).
        """
        logger.info(f"{'='*60}")
        logger.info(f"  Vantage Engine v4.1 | {self.scan_id}")
        logger.info(f"{'='*60}")

        # ── Layer 1: Structured fetch ──────────────────────────────────────────
        logger.info("[ LAYER 1 ] Structured API fetch...")
        raw_df = self.fetcher.fetch_all()
        logger.info(f"  Raw records: {len(raw_df)}")

        # ── Normalize ──────────────────────────────────────────────────────────
        logger.info("[ NORMALIZE ] Canonical names + kickoff format...")
        norm_df = normalize_dataframe(raw_df) if not raw_df.empty else raw_df

        # ── Odds enrichment ────────────────────────────────────────────────────
        logger.info("[ ODDS ] Fetching odds...")
        enriched_df = self.odds.enrich_with_odds(norm_df) if not norm_df.empty else norm_df

        # ── Fallback (if needed) ───────────────────────────────────────────────
        fallback_df = self.fallback.fetch(
            min_structured_count=self.MIN_STRUCTURED_MATCHES,
            structured_df=enriched_df,
        )
        if not fallback_df.empty:
            logger.info(f"[ FALLBACK ] Adding {len(fallback_df)} scrape matches")
            fallback_norm = normalize_dataframe(fallback_df)
            enriched_df = pd.concat([enriched_df, fallback_norm], ignore_index=True)

        # ── Validate ───────────────────────────────────────────────────────────
        logger.info("[ VALIDATE ] Filtering and scoring...")
        clean_df = self.validator.validate(enriched_df) if not enriched_df.empty else enriched_df

        if clean_df.empty:
            logger.critical("Pipeline produced 0 valid matches. Check API keys and sources.")
            return pd.DataFrame()

        # ── Report ─────────────────────────────────────────────────────────────
        print(self.validator.report(clean_df))

        # ── Save ───────────────────────────────────────────────────────────────
        ts       = self.scan_time.strftime("%Y%m%d_%H%M")
        csv_path = self.output_dir / f"harvest_{ts}.csv"
        json_path = self.output_dir / f"harvest_{ts}.json"

        clean_df.to_csv(csv_path, index=False)
        clean_df.to_json(json_path, orient="records", date_format="iso", indent=2)
        logger.info(f"Saved: {csv_path} | {json_path}")
        logger.info(f"Final: {len(clean_df)} matches | {len(clean_df.columns)} columns each")

        return clean_df

    def preview(self, df: pd.DataFrame, n: int = 15):
        """Print a clean preview of the harvest."""
        if df.empty:
            print("Empty — nothing to preview")
            return

        display_cols = [c for c in [
            "kickoff_utc", "league", "home", "away",
            "odds_1", "odds_X", "odds_2", "odds_over25",
            "has_odds", "confidence", "source"
        ] if c in df.columns]

        print(f"\n{'─'*80}")
        print(f"  HARVEST PREVIEW — {len(df)} matches")
        print(f"{'─'*80}")
        print(df[display_cols].head(n).to_string(index=False))
        print(f"{'─'*80}\n")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    engine = VantageEngine()
    df = engine.run()
    if not df.empty:
        engine.preview(df)
    else:
        print("\n❌ No matches harvested. Check vantage_engine.log\n")
        print("Quick fixes:")
        print("  1. Get a free API-Football key at api-sports.io (100 req/day)")
        print("  2. Get a free The Odds API key at the-odds-api.com (500 req/month)")
        print("  3. Set keys in engine.py or as env vars API_FOOTBALL_KEY / THE_ODDS_API_KEY")

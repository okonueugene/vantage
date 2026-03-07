"""
validator.py
────────────
Cross-validates fixtures from multiple sources.

Logic:
  - A match confirmed by 2+ sources → HIGH confidence
  - A match from 1 source only     → MEDIUM (kept, flagged)
  - Kickoff outside 24h window     → DROPPED
  - No valid team names             → DROPPED
  - Duplicate canonical IDs        → MERGED (best source wins)

Output: clean DataFrame with confidence scores, ready for enrichment.
"""

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import List

import pandas as pd

logger = logging.getLogger("VantageV5.Validator")

# Source priority for deduplication (higher = preferred)
SOURCE_PRIORITY = {
    "API-Football": 3,
    "ESPN_API":     2,
    "TheOddsAPI":   2,
    "TheSportsDB":  1,
    "Soccerway":    1,
    "ESPN":         1,
}

# Junk strings — team names that are NOT team names
JUNK_STRINGS = {
    "terms of use", "privacy policy", "cookie policy", "contact us",
    "sign in", "log in", "register", "subscribe", "newsletter",
    "soccer scores", "choose a league", "favorites", "matches",
    "live scores", "highlights", "news", "standings", "table",
    "all leagues", "settings", "language", "home", "away",
    "score", "result", "fixture", "upcoming", "loading",
    "espn+", "espn deportes", "espn fc", "sky sports",
    "usa net", "universo", "football", "soccer", "hockey",
}


def _is_valid_team(name: str) -> bool:
    if not name or len(name) < 3 or len(name) > 45:
        return False
    if name.lower().strip() in JUNK_STRINGS:
        return False
    if re.search(r'https?://|www\.', name):
        return False
    if re.search(r'\+\d*$|sport\s*\d+$', name, re.I):
        return False
    if len(re.findall(r'[A-Za-zÀ-ÿ]', name)) < 2:
        return False
    return True


def _within_24h(kickoff_str: str) -> bool:
    try:
        if not kickoff_str:
            return True   # include if no timestamp
        # Handles "2026-03-01 15:00 UTC" and ISO formats
        kickoff_str = kickoff_str.replace(" UTC", "+00:00").replace("Z", "+00:00")
        if "T" not in kickoff_str and " " in kickoff_str:
            kickoff_str = kickoff_str.replace(" ", "T", 1)
        dt  = datetime.fromisoformat(kickoff_str)
        now = datetime.now(timezone.utc)
        return now - timedelta(hours=1) <= dt <= now + timedelta(hours=25)
    except Exception:
        return True


class Validator:

    def __init__(self, require_odds: bool = False, min_confidence: str = "low"):
        """
        Args:
            require_odds:    If True, drop matches with no odds at all.
            min_confidence:  'high' = multi-source only, 'medium' = 1+ source, 'low' = all valid
        """
        self.require_odds    = require_odds
        self.min_confidence  = min_confidence

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full validation pipeline:
          1. Drop invalid team names
          2. Drop out-of-window kickoffs
          3. Merge duplicates by canonical_id
          4. Score confidence
          5. Apply minimum confidence filter
        """
        if df.empty:
            logger.warning("Validator received empty DataFrame")
            return df

        original_count = len(df)

        # ── Step 1: Drop invalid teams ─────────────────────────────────────────
        valid_teams = df["home"].apply(_is_valid_team) & df["away"].apply(_is_valid_team)
        df = df[valid_teams].copy()
        logger.info(f"Team filter: {original_count} → {len(df)} (dropped {original_count - len(df)} junk rows)")

        # ── Step 2: Drop out-of-window kickoffs ────────────────────────────────
        if "kickoff_utc" in df.columns:
            in_window = df["kickoff_utc"].apply(_within_24h)
            before = len(df)
            df = df[in_window].copy()
            logger.info(f"Time filter: {before} → {len(df)} (dropped {before - len(df)} out-of-window)")

        if df.empty:
            logger.warning("No valid matches after filtering")
            return df

        # ── Step 3: Deduplicate by canonical_id ────────────────────────────────
        if "canonical_id" not in df.columns:
            # Generate if missing
            df["canonical_id"] = df.apply(
                lambda r: f"{re.sub(r'[^A-Za-z]', '', str(r['home']))[:4].upper()}"
                          f"_{re.sub(r'[^A-Za-z]', '', str(r['away']))[:4].upper()}",
                axis=1
            )

        # Count how many sources confirmed each match
        source_counts = df.groupby("canonical_id")["source"].nunique()
        df["source_count"] = df["canonical_id"].map(source_counts)

        # For each canonical_id, keep the record from the highest-priority source
        df["source_priority"] = df["source"].map(SOURCE_PRIORITY).fillna(0)
        df = df.sort_values("source_priority", ascending=False)
        df = df.drop_duplicates(subset=["canonical_id"], keep="first")
        df = df.drop(columns=["source_priority"])

        # ── Step 4: Confidence scoring ─────────────────────────────────────────
        df["confidence"] = df["source_count"].apply(
            lambda n: "high" if n >= 2 else "medium" if n == 1 else "low"
        )

        # Boost confidence if odds are present
        has_odds_col = "has_odds" if "has_odds" in df.columns else None
        if has_odds_col:
            df.loc[df[has_odds_col] & (df["confidence"] == "medium"), "confidence"] = "high"

        # ── Step 5: Apply minimum confidence filter ────────────────────────────
        confidence_order = {"low": 0, "medium": 1, "high": 2}
        min_level = confidence_order.get(self.min_confidence, 0)
        df = df[df["confidence"].map(confidence_order) >= min_level]

        # ── Step 6: Optional odds filter ──────────────────────────────────────
        if self.require_odds and has_odds_col:
            before = len(df)
            df = df[df[has_odds_col]]
            logger.info(f"Odds filter: {before} → {len(df)} (require_odds=True)")

        # ── Final sort ─────────────────────────────────────────────────────────
        sort_col = "kickoff_utc" if "kickoff_utc" in df.columns else "canonical_id"
        df = df.sort_values(sort_col).reset_index(drop=True)

        logger.info(
            f"Validation complete: {len(df)} clean matches | "
            f"High: {(df['confidence']=='high').sum()} | "
            f"Medium: {(df['confidence']=='medium').sum()}"
        )

        return df

    def report(self, df: pd.DataFrame) -> str:
        """Print a quick diagnostic summary."""
        if df.empty:
            return "Empty DataFrame — no matches to report"

        lines = [
            f"\n{'═'*55}",
            f"  VALIDATION REPORT",
            f"{'═'*55}",
            f"  Total matches:   {len(df)}",
            f"  High confidence: {(df.get('confidence','') == 'high').sum()}",
            f"  With odds:       {df.get('has_odds', pd.Series([False]*len(df))).sum()}",
            f"  Leagues covered: {df['league'].nunique()}",
            f"{'─'*55}",
        ]
        for league in sorted(df["league"].unique()):
            subset = df[df["league"] == league]
            odds   = subset.get("has_odds", pd.Series([False]*len(subset))).sum()
            lines.append(f"  {league:<30} {len(subset):>3} matches  {odds:>3} with odds")

        lines.append(f"{'═'*55}\n")
        return "\n".join(lines)

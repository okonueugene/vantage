"""
structured_fetcher.py
─────────────────────
Layer 1: Structured JSON APIs — primary fixture discovery.

Sources (in priority order):
  1. ESPN Site API       — free, no key, stable, great coverage
  2. TheSportsDB         — free tier, good metadata
  3. API-Football        — free tier (100 req/day), best odds data

Never scrapes HTML. Never uses BeautifulSoup. Pure JSON.
"""

import time
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import requests
import pandas as pd

logger = logging.getLogger("VantageV5.Fetcher")

# ──────────────────────────────────────────────
# League Registry
# Only these leagues are in scope. Add/remove freely.
# ──────────────────────────────────────────────
ESPN_LEAGUES = {
    "eng.1":  "Premier League",
    "esp.1":  "La Liga",
    "ger.1":  "Bundesliga",
    "ita.1":  "Serie A",
    "fra.1":  "Ligue 1",
    "ned.1":  "Eredivisie",
    "por.1":  "Primeira Liga",
    "tur.1":  "Süper Lig",
    "uefa.champions": "Champions League",
    "uefa.europa":    "Europa League",
}

THESPORTSDB_LEAGUES = {
    "4328": "Premier League",
    "4335": "La Liga",
    "4331": "Bundesliga",
    "4332": "Serie A",
    "4334": "Ligue 1",
    "4337": "Eredivisie",
}

# Set your API-Football key here (or load from env)
API_FOOTBALL_KEY = ""   # os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_LEAGUES = {
    39:  "Premier League",
    140: "La Liga",
    78:  "Bundesliga",
    135: "Serie A",
    61:  "Ligue 1",
    2:   "Champions League",
    3:   "Europa League",
}


class StructuredFetcher:

    ESPN_BASE    = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
    TSDB_BASE    = "https://www.thesportsdb.com/api/v1/json/3/eventsnextleague.php"
    APIF_BASE    = "https://v3.football.api-sports.io/fixtures"

    def __init__(self, api_football_key: str = API_FOOTBALL_KEY):
        self.api_football_key = api_football_key
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (compatible; VantageEngine/5.0)",
            "Accept": "application/json",
        })
        self.today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _get(self, url: str, params: Dict = None, headers: Dict = None, delay: float = None) -> Optional[Dict]:
        """Single GET with minimal delay (structured APIs don't need stealth)."""
        if delay is None:
            delay = random.uniform(0.8, 1.6)
        time.sleep(delay)
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"GET failed [{url}]: {e}")
            return None

    # ──────────────────────────────────────────────
    # Source 1: ESPN Site API
    # ──────────────────────────────────────────────
    def fetch_espn(self) -> List[Dict]:
        """
        Iterates ESPN league codes and returns today's + next 24h fixtures.
        No auth required. Returns clean structured JSON.
        """
        records = []
        for code, league_name in ESPN_LEAGUES.items():
            url  = self.ESPN_BASE.format(league=code)
            data = self._get(url)
            if not data:
                continue

            for event in data.get("events", []):
                try:
                    comp       = event["competitions"][0]
                    competitors = comp.get("competitors", [])
                    if len(competitors) < 2:
                        continue

                    # ESPN always returns home first (index 0)
                    home_obj = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                    away_obj = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

                    home    = home_obj["team"]["displayName"]
                    away    = away_obj["team"]["displayName"]
                    kickoff = event.get("date", "")           # ISO 8601 UTC string

                    # Filter: only next 24 hours
                    if not self._within_24h(kickoff):
                        continue

                    # Status
                    status = comp.get("status", {}).get("type", {}).get("name", "scheduled")

                    records.append({
                        "match_id":    f"ESPN_{event['id']}",
                        "league":      league_name,
                        "kickoff_utc": kickoff,
                        "home":        home,
                        "away":        away,
                        "status":      status,
                        "source":      "ESPN_API",
                    })
                except (KeyError, IndexError) as e:
                    logger.debug(f"ESPN parse error in {league_name}: {e}")

            logger.info(f"ESPN {league_name}: {len([r for r in records if r.get('league') == league_name])} matches")

        return records

    # ──────────────────────────────────────────────
    # Source 2: TheSportsDB
    # ──────────────────────────────────────────────
    def fetch_thesportsdb(self) -> List[Dict]:
        """
        TheSportsDB free tier — next events per league.
        Good for cross-validation of team names and kickoff times.
        """
        records = []
        for league_id, league_name in THESPORTSDB_LEAGUES.items():
            data = self._get(self.TSDB_BASE, params={"id": league_id})
            if not data:
                continue

            for event in (data.get("events") or []):
                try:
                    kickoff = f"{event.get('dateEvent', '')}T{event.get('strTime', '00:00:00')}Z"
                    if not self._within_24h(kickoff):
                        continue

                    records.append({
                        "match_id":    f"TSDB_{event['idEvent']}",
                        "league":      league_name,
                        "kickoff_utc": kickoff,
                        "home":        event["strHomeTeam"],
                        "away":        event["strAwayTeam"],
                        "status":      "scheduled",
                        "source":      "TheSportsDB",
                    })
                except KeyError as e:
                    logger.debug(f"TSDB parse error: {e}")

        logger.info(f"TheSportsDB: {len(records)} matches across {len(THESPORTSDB_LEAGUES)} leagues")
        return records

    # ──────────────────────────────────────────────
    # Source 3: API-Football (requires free key)
    # ──────────────────────────────────────────────
    def fetch_api_football(self) -> List[Dict]:
        """
        API-Football free tier — 100 requests/day.
        Best source for odds + fixture metadata.
        Set API_FOOTBALL_KEY at top of file or via env var.
        """
        if not self.api_football_key:
            logger.info("API-Football key not set — skipping (set API_FOOTBALL_KEY)")
            return []

        headers = {
            "x-rapidapi-key":  self.api_football_key,
            "x-rapidapi-host": "v3.football.api-sports.io",
        }

        records = []
        for league_id, league_name in API_FOOTBALL_LEAGUES.items():
            data = self._get(
                self.APIF_BASE,
                params={"league": league_id, "date": self.today, "season": 2025},
                headers=headers,
            )
            if not data:
                continue

            for fx in (data.get("response") or []):
                try:
                    fixture = fx["fixture"]
                    teams   = fx["teams"]
                    goals   = fx.get("goals", {})

                    records.append({
                        "match_id":    f"APIF_{fixture['id']}",
                        "league":      league_name,
                        "kickoff_utc": fixture["date"],
                        "home":        teams["home"]["name"],
                        "away":        teams["away"]["name"],
                        "status":      fixture["status"]["short"],
                        "venue":       fixture.get("venue", {}).get("name", ""),
                        "referee":     fixture.get("referee", ""),
                        "source":      "API-Football",
                    })
                except KeyError as e:
                    logger.debug(f"API-Football parse error: {e}")

        logger.info(f"API-Football: {len(records)} matches")
        return records

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────
    def _within_24h(self, kickoff_iso: str) -> bool:
        """True if kickoff falls within the next 24 hours from now."""
        try:
            if not kickoff_iso:
                return False
            # Handle both "2026-03-01T15:00:00Z" and "2026-03-01T15:00:00+00:00"
            kickoff_iso = kickoff_iso.replace("Z", "+00:00")
            kickoff_dt  = datetime.fromisoformat(kickoff_iso)
            now         = datetime.now(timezone.utc)
            return now <= kickoff_dt <= now + timedelta(hours=24)
        except Exception:
            return True   # Default include if unparseable

    # ──────────────────────────────────────────────
    # Main: fetch all structured sources → DataFrame
    # ──────────────────────────────────────────────
    def fetch_all(self) -> pd.DataFrame:
        """
        Runs all three structured sources, merges results.
        Priority: ESPN_API > API-Football > TheSportsDB
        """
        logger.info("Layer 1: Fetching structured APIs...")

        espn_records = self.fetch_espn()
        apif_records = self.fetch_api_football()
        tsdb_records = self.fetch_thesportsdb()

        all_records = espn_records + apif_records + tsdb_records

        if not all_records:
            logger.warning("All structured sources returned empty")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # Ensure required columns exist
        for col in ["match_id", "league", "kickoff_utc", "home", "away", "status", "source"]:
            if col not in df.columns:
                df[col] = "N/A"

        logger.info(
            f"Layer 1 complete — {len(df)} raw records "
            f"(ESPN: {len(espn_records)}, APIF: {len(apif_records)}, TSDB: {len(tsdb_records)})"
        )
        return df

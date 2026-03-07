"""
odds_fetcher.py
───────────────
Layer 1b: Odds enrichment from structured sources.

Priority:
  1. API-Football odds endpoint (best, requires free key)
  2. The Odds API       (generous free tier — 500 req/month)
  3. Betfair BSP proxy  (public endpoint, no auth needed for some markets)

Odds are fetched AFTER fixtures are known — targeted calls only.
Never bulk-scrapes odds pages.
"""

import time
import logging
import random
from typing import Dict, List, Optional

import requests
import pandas as pd

logger = logging.getLogger("VantageV5.OddsFetcher")

# Set keys here or load from environment variables
API_FOOTBALL_KEY = ""   # os.getenv("API_FOOTBALL_KEY")
THE_ODDS_API_KEY = ""   # os.getenv("THE_ODDS_API_KEY") — free at the-odds-api.com

# Markets to pull (The Odds API market codes)
ODDS_MARKETS = "h2h,totals"   # 1X2 + over/under


class OddsFetcher:

    APIF_ODDS_URL  = "https://v3.football.api-sports.io/odds"
    THEODDS_URL    = "https://api.the-odds-api.com/v4/sports/soccer_{sport_key}/odds"
    BETFAIR_URL    = "https://api.betfair.com/exchange/betting/json-rpc/v1"

    # Map our league names to The Odds API sport keys
    THEODDS_SPORT_KEYS = {
        "Premier League":   "epl",
        "La Liga":          "spain_la_liga",
        "Bundesliga":       "germany_bundesliga",
        "Serie A":          "italy_serie_a",
        "Ligue 1":          "france_ligue_1",
        "Champions League": "soccer_uefa_champs_league",
        "Europa League":    "soccer_uefa_europa_league",
    }

    def __init__(self, api_football_key: str = API_FOOTBALL_KEY,
                 the_odds_key: str = THE_ODDS_API_KEY):
        self.apif_key    = api_football_key
        self.theodds_key = the_odds_key
        self.session     = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "VantageEngine/5.0",
        })

    def _get(self, url: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        time.sleep(random.uniform(0.8, 1.6))
        try:
            resp = self.session.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Odds GET failed [{url}]: {e}")
            return None

    # ──────────────────────────────────────────────
    # Source 1: API-Football Odds
    # ──────────────────────────────────────────────
    def fetch_apif_odds(self, fixture_id: str) -> Dict:
        """
        Fetch odds for a specific fixture by API-Football fixture ID.
        Call this after structured_fetcher has identified matches.
        Free tier: 100 calls/day — use sparingly, top leagues only.
        """
        if not self.apif_key:
            return {}

        data = self._get(
            self.APIF_ODDS_URL,
            params={"fixture": fixture_id.replace("APIF_", ""), "bookmaker": 6},  # Bet365
            headers={
                "x-rapidapi-key":  self.apif_key,
                "x-rapidapi-host": "v3.football.api-sports.io",
            }
        )
        if not data or not data.get("response"):
            return {}

        try:
            bets = data["response"][0]["bookmakers"][0]["bets"]
            result = {}

            for bet in bets:
                name = bet["name"].lower()
                values = {v["value"]: float(v["odd"]) for v in bet["values"]}

                if "match winner" in name:
                    result["1"] = values.get("Home", "-")
                    result["X"] = values.get("Draw", "-")
                    result["2"] = values.get("Away", "-")
                elif "goals over/under" in name:
                    result["over25"]  = values.get("Over 2.5", "-")
                    result["under25"] = values.get("Under 2.5", "-")

            return result
        except (KeyError, IndexError):
            return {}

    # ──────────────────────────────────────────────
    # Source 2: The Odds API (recommended — generous free tier)
    # ──────────────────────────────────────────────
    def fetch_theodds(self, league_name: str) -> List[Dict]:
        """
        Fetch all upcoming odds for a league from The Odds API.
        Returns list of {home, away, odds} dicts.
        Free tier: 500 requests/month. Sign up at the-odds-api.com.
        """
        if not self.theodds_key:
            return []

        sport_key = self.THEODDS_SPORT_KEYS.get(league_name)
        if not sport_key:
            return []

        url  = self.THEODDS_URL.format(sport_key=sport_key)
        data = self._get(url, params={
            "apiKey":  self.theodds_key,
            "regions": "uk",           # uk bookmakers (Bet365, Betfair, etc.)
            "markets": ODDS_MARKETS,
            "oddsFormat": "decimal",
        })

        if not isinstance(data, list):
            return []

        results = []
        for match in data:
            try:
                home = match["home_team"]
                away = match["away_team"]
                odds = self._parse_theodds_bookmakers(match.get("bookmakers", []))
                if odds:
                    results.append({"home": home, "away": away, "odds": odds})
            except KeyError:
                continue

        logger.info(f"The Odds API → {league_name}: {len(results)} matches with odds")
        return results

    def _parse_theodds_bookmakers(self, bookmakers: List[Dict]) -> Dict:
        """Extract best available odds from The Odds API bookmaker list."""
        # Prefer Bet365 (key: bet365), fallback to first available
        preferred = ["bet365", "betfair", "paddypower", "williamhill"]

        for pref in preferred:
            bm = next((b for b in bookmakers if b.get("key") == pref), None)
            if bm:
                return self._extract_odds_from_markets(bm.get("markets", []))

        # Fallback: first bookmaker
        if bookmakers:
            return self._extract_odds_from_markets(bookmakers[0].get("markets", []))
        return {}

    def _extract_odds_from_markets(self, markets: List[Dict]) -> Dict:
        result = {}
        for market in markets:
            key = market.get("key", "")
            outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}

            if key == "h2h":
                # The Odds API uses full team names for outcomes
                vals = list(outcomes.values())
                names = list(outcomes.keys())
                if len(vals) == 3:
                    result["1"] = vals[0]
                    result["X"] = vals[1]
                    result["2"] = vals[2]
                elif len(vals) == 2:
                    result["1"] = vals[0]
                    result["2"] = vals[1]

            elif key == "totals":
                result["over25"]  = outcomes.get("Over",  "-")
                result["under25"] = outcomes.get("Under", "-")

        return result

    # ──────────────────────────────────────────────
    # Main: enrich a fixtures DataFrame with odds
    # ──────────────────────────────────────────────
    def enrich_with_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes the normalized fixtures DataFrame and adds odds columns.
        Strategy:
          - If API-Football IDs present and key set → use APIF odds (per-fixture)
          - If The Odds API key set → bulk-fetch per league, then match by team name
          - Otherwise: odds columns filled with '-'
        """
        if df.empty:
            return df

        # Initialize odds columns
        df["odds_1"]      = "-"
        df["odds_X"]      = "-"
        df["odds_2"]      = "-"
        df["odds_over25"] = "-"
        df["odds_u25"]    = "-"
        df["odds_source"] = "none"

        # ── Strategy A: The Odds API (bulk per league) ─────────────────────────
        if self.theodds_key:
            for league_name in df["league"].unique():
                league_odds = self.fetch_theodds(league_name)
                if not league_odds:
                    continue

                for lo in league_odds:
                    # Match by team name (normalised lowercase)
                    mask = (
                        df["home"].str.lower().str.contains(lo["home"].lower()[:8], na=False) &
                        df["away"].str.lower().str.contains(lo["away"].lower()[:8], na=False)
                    )
                    if mask.any():
                        o = lo["odds"]
                        df.loc[mask, "odds_1"]      = o.get("1", "-")
                        df.loc[mask, "odds_X"]      = o.get("X", "-")
                        df.loc[mask, "odds_2"]      = o.get("2", "-")
                        df.loc[mask, "odds_over25"] = o.get("over25", "-")
                        df.loc[mask, "odds_u25"]    = o.get("under25", "-")
                        df.loc[mask, "odds_source"] = "TheOddsAPI"

        # ── Strategy B: API-Football per-fixture (uses remaining quota) ────────
        elif self.apif_key:
            apif_rows = df[df["source"] == "API-Football"]
            for idx, row in apif_rows.iterrows():
                odds = self.fetch_apif_odds(row["match_id"])
                if odds:
                    df.at[idx, "odds_1"]      = odds.get("1", "-")
                    df.at[idx, "odds_X"]      = odds.get("X", "-")
                    df.at[idx, "odds_2"]      = odds.get("2", "-")
                    df.at[idx, "odds_over25"] = odds.get("over25", "-")
                    df.at[idx, "odds_u25"]    = odds.get("under25", "-")
                    df.at[idx, "odds_source"] = "API-Football"

        # Implied probability columns
        df["has_odds"] = df["odds_1"].apply(lambda x: x != "-")
        df.loc[df["has_odds"], "implied_home"] = df.loc[df["has_odds"], "odds_1"].apply(
            lambda x: round(1 / float(x), 4) if x != "-" else None
        )
        df.loc[df["has_odds"], "implied_over25"] = df.loc[df["has_odds"], "odds_over25"].apply(
            lambda x: round(1 / float(x), 4) if x not in ("-", None) else None
        )

        odds_count = df["has_odds"].sum()
        logger.info(f"Odds enrichment complete — {odds_count}/{len(df)} matches have odds")
        return df

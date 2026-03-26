"""
odds_fetcher.py
───────────────
Odds enrichment using **only** API-Football[](https://v3.football.api-sports.io/)

Uses the working auth headers from your test:
  - x-rapidapi-key
  - x-rapidapi-host: "v3.football.api-sports.io"

.env must have:
  API_FOOTBALL_KEY=your_working_key_here

Strategy:
  - Bulk fetch /odds?date=... for relevant dates
  - Parse bookmaker bets (prefer Bet365 id=8)
  - Fuzzy-match team names to your fixture rows
"""

import time
import logging
import random
import os
from datetime import date
from difflib import SequenceMatcher
from dotenv import load_dotenv
from typing import Dict, List, Optional

import requests
import pandas as pd

load_dotenv()
logger = logging.getLogger("VantageV5.OddsFetcher")

# Bookmaker priority (Bet365=8 is usually highest quality)
PREFERRED_BOOKMAKERS = [8, 4, 16, 11, 6, 7, 1]

# Bet IDs we extract
BET_MATCH_WINNER  = 1    # 1X2
BET_GOALS_OU      = 5    # Over/Under (2.5 line)
BET_BTTS          = 8    # Both Teams To Score
BET_CORNERS_TOTAL = 45   # Total Corners
BET_CARDS_TOTAL   = 56   # Total Cards/Bookings

FUZZY_THRESHOLD = 0.60   # Tune: lower = more matches, higher = stricter


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _best_bookmaker_bets(bookmakers: List[Dict]) -> List[Dict]:
    """Return bets from the highest-priority bookmaker available"""
    for bm_id in PREFERRED_BOOKMAKERS:
        for bm in bookmakers:
            if bm.get("id") == bm_id:
                return bm.get("bets", [])
    return bookmakers[0].get("bets", []) if bookmakers else []


def _parse_bets(bets: List[Dict]) -> Dict:
    """Extract relevant odds from bets list"""
    result = {}
    for bet in bets:
        bid = bet.get("id")
        vals = {
            v["value"]: float(v["odd"])
            for v in bet.get("values", [])
            if v.get("odd") not in (None, "", "-")
        }

        if bid == BET_MATCH_WINNER:
            result["1"] = vals.get("Home", "-")
            result["X"] = vals.get("Draw", "-")
            result["2"] = vals.get("Away", "-")

        elif bid == BET_GOALS_OU:
            result["over25"]  = vals.get("Over 2.5", "-")
            result["under25"] = vals.get("Under 2.5", "-")

        elif bid == BET_BTTS:
            result["btts_yes"] = vals.get("Yes", "-")
            result["btts_no"]  = vals.get("No", "-")

        elif bid == BET_CORNERS_TOTAL:
            for label, odd in vals.items():
                if str(label).startswith("Over"):
                    result["corners_over"] = odd
                    break

        elif bid == BET_CARDS_TOTAL:
            for label, odd in vals.items():
                if str(label).startswith("Over"):
                    result["cards_over"] = odd
                    break

    return result


class OddsFetcher:

    BASE = "https://v3.football.api-sports.io"

    def __init__(self):
        key = os.getenv("API_FOOTBALL_KEY", "").strip()
        if not key:
            logger.error("No API_FOOTBALL_KEY in .env — odds enrichment disabled")
            self.enabled = False
            return

        self.enabled = True
        self.session = requests.Session()
        self.session.headers.update({
            "x-rapidapi-key":  key,
            "x-rapidapi-host": "v3.football.api-sports.io",
            "Accept":          "application/json",
            "User-Agent":      "VantageEngine/5.0",
        })

        # Cache per date
        self._odds_cache: Dict[str, List[Dict]] = {}

    def _get(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        if not self.enabled:
            return None

        url = f"{self.BASE}/{endpoint}"
        time.sleep(random.uniform(0.7, 1.4))

        try:
            resp = self.session.get(url, params=params, timeout=12)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.warning(f"API-Football HTTP {e.response.status_code}: {url}")
            return None
        except Exception as e:
            logger.warning(f"API-Football request failed: {e}")
            return None

    def _load_fixtures_for_date(self, date_str: str) -> Dict[int, Dict]:
        """
        GET /fixtures?date=  →  {fixture_id: {home, away, kickoff_ts}}
        The /odds endpoint has NO team names — we must join from here.
        """
        data = self._get("fixtures", {"date": date_str})
        result: Dict[int, Dict] = {}
        if not data or "response" not in data:
            return result
        for rec in data["response"]:
            fid   = rec.get("fixture", {}).get("id")
            teams = rec.get("teams", {})
            if not fid:
                continue
            result[fid] = {
                "home":       teams.get("home", {}).get("name", ""),
                "away":       teams.get("away", {}).get("name", ""),
                "kickoff_ts": rec.get("fixture", {}).get("timestamp"),
            }
        logger.info(f"APIF /fixtures?date={date_str} → {len(result)} fixtures")
        return result

    def _load_odds_for_date(self, date_str: str) -> List[Dict]:
        """
        Two-call strategy (discovered 2026-03-08):
          1. /fixtures?date=  → fixture_id → home/away names
          2. /odds?date=      → fixture_id → bookmaker odds
          JOIN on fixture_id to produce enriched records with team names.
        """
        if date_str in self._odds_cache:
            return self._odds_cache[date_str]

        # Call 1: team names
        fixture_map = self._load_fixtures_for_date(date_str)

        # Call 2: odds
        data = self._get("odds", {"date": date_str})
        if not data or "response" not in data:
            self._odds_cache[date_str] = []
            return []

        odds_list = []
        for entry in data["response"]:
            fid = entry.get("fixture", {}).get("id")
            bookmakers = entry.get("bookmakers", [])
            if not fid or not bookmakers:
                continue

            bets   = _best_bookmaker_bets(bookmakers)
            parsed = _parse_bets(bets)
            if not parsed:
                continue

            # Join team names from /fixtures (not available in /odds response)
            fix_info = fixture_map.get(fid, {})
            odds_list.append({
                "fixture_id": fid,
                "home":       fix_info.get("home", ""),
                "away":       fix_info.get("away", ""),
                "kickoff_ts": fix_info.get("kickoff_ts"),
                **parsed
            })

        self._odds_cache[date_str] = odds_list
        logger.info(f"APIF joined {len(odds_list)} fixtures+odds for {date_str}")
        return odds_list

    def enrich_with_odds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich df with odds from API-Football.
        Expects columns: home, away, kickoff (datetime/ISO string), optionally league
        """
        if df.empty or not self.enabled:
            return df

        # Ensure output columns — use object dtype to accept both str sentinel ("-")
        # and float odds values without pandas dtype coercion errors.
        ODDS_COLS = ["odds_1", "odds_X", "odds_2", "odds_over25", "odds_u25",
                     "odds_btts_yes", "odds_corners_over", "odds_cards_over"]
        for col in ODDS_COLS:
            if col not in df.columns:
                df[col] = pd.Series(["-"] * len(df), index=df.index, dtype=object)
            else:
                df[col] = df[col].astype(object)
        if "odds_source" not in df.columns:
            df["odds_source"] = pd.Series(["none"] * len(df), index=df.index, dtype=object)
        else:
            df["odds_source"] = df["odds_source"].astype(object)

        # Collect unique dates
        if "kickoff" in df.columns:
            dates = df["kickoff"].dropna().astype(str).str[:10].unique().tolist()
        else:
            dates = [str(date.today())]

        # Pre-load all dates
        for d in dates:
            self._load_odds_for_date(d)

        enriched = 0
        for idx, row in df.iterrows():
            if row["odds_source"] != "none":
                continue

            ds = str(row.get("kickoff", date.today()))[:10]
            records = self._odds_cache.get(ds, [])
            if not records:
                continue

            # Fuzzy match by team names
            best = None
            best_score = 0.0

            for rec in records:
                h_sim = _similarity(row.get("home", ""), rec["home"])
                a_sim = _similarity(row.get("away", ""), rec["away"])
                score = (h_sim + a_sim) / 2

                if score > best_score and score >= FUZZY_THRESHOLD:
                    best_score = score
                    best = rec

            if best:
                def _s(v):
                    """Return float if numeric, else sentinel string."""
                    return float(v) if v not in ("-", None, "") else "-"
                df.at[idx, "odds_1"]            = _s(best.get("1"))
                df.at[idx, "odds_X"]            = _s(best.get("X"))
                df.at[idx, "odds_2"]            = _s(best.get("2"))
                df.at[idx, "odds_over25"]       = _s(best.get("over25"))
                df.at[idx, "odds_u25"]          = _s(best.get("under25"))
                df.at[idx, "odds_btts_yes"]     = _s(best.get("btts_yes"))
                df.at[idx, "odds_corners_over"] = _s(best.get("corners_over"))
                df.at[idx, "odds_cards_over"]   = _s(best.get("cards_over"))
                df.at[idx, "odds_source"]       = "API-Football"
                enriched += 1
                logger.info(f"Matched: {row.get('home')} vs {row.get('away')} → score={best_score:.3f}")
            else:
                logger.debug(f"No match: {row.get('home')} vs {row.get('away')} (best={best_score:.3f})")

        total = (df["odds_1"] != "-").sum()
        logger.info(f"API-Football enrichment complete — {enriched} new, {total}/{len(df)} total matches have odds")
        return df

# ─────────────────────────────────────────────────────────────────────────────
# TheOddsAPI fallback — bulk per-league, covers EPL/La Liga/Bundesliga/etc.
# Free tier: 500 requests/month.  Add ODDS_API_KEY to .env to enable.
# ─────────────────────────────────────────────────────────────────────────────

THEODDS_SPORT_KEYS = {
    # Top 5
    "Premier League":        "soccer_england_league1",   # mapped to EPL
    "La Liga":               "soccer_spain_la_liga",
    "Bundesliga":            "soccer_germany_bundesliga",
    "Serie A":               "soccer_italy_serie_a",
    "Ligue 1":               "soccer_france_ligue_one",
    # Secondary European
    "Championship":          "soccer_england_league1",
    "Scottish Premiership":  "soccer_scotland_premiership",
    "Eredivisie":            "soccer_netherlands_eredivisie",
    "Primeira Liga":         "soccer_portugal_primeira_liga",
    "Süper Lig":             "soccer_turkey_super_lig",
    "Belgian Pro League":    "soccer_belgium_first_div",
    "Greek Super League":    "soccer_greece_super_league",
    # Cups
    "Champions League":      "soccer_uefa_champs_league",
    "Europa League":         "soccer_uefa_europa_league",
    "Conference League":     "soccer_uefa_europa_conference_league",
    # Americas
    "MLS":                   "soccer_usa_mls",
    "Liga MX":               "soccer_mexico_ligamx",
    "Brasileirao":           "soccer_brazil_campeonato",
    "Argentine Primera":     "soccer_argentina_primera_division",
}

THEODDS_BASE = "https://api.the-odds-api.com/v4"


class TheOddsAPIFetcher:
    """
    Fallback odds source for leagues not covered by APIF free plan.
    Uses bulk /sports/{key}/odds endpoint — one call per league covers all matches.
    """

    def __init__(self):
        self.key = os.getenv("ODDS_API_KEY", "").strip()
        self.enabled = bool(self.key)
        if not self.enabled:
            logger.debug("No ODDS_API_KEY — TheOddsAPI fallback disabled")

        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json",
                                     "User-Agent": "VantageEngine/5.0"})
        self._cache: Dict[str, List[Dict]] = {}   # sport_key → records

    def _get(self, url: str, params: Dict) -> Optional[object]:
        time.sleep(random.uniform(0.5, 1.0))
        try:
            resp = self.session.get(url, params=params, timeout=12)
            resp.raise_for_status()
            remaining = resp.headers.get("x-requests-remaining", "?")
            logger.debug(f"TheOddsAPI remaining={remaining}")
            return resp.json()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status == 401:
                logger.error("TheOddsAPI 401 — check ODDS_API_KEY")
            elif status == 422:
                logger.warning(f"TheOddsAPI 422 — bad params: {url}")
            else:
                logger.warning(f"TheOddsAPI HTTP {status}: {url}")
            return None
        except Exception as e:
            logger.warning(f"TheOddsAPI request failed: {e}")
            return None

    def _load_league(self, sport_key: str) -> List[Dict]:
        """Bulk fetch all upcoming matches for one sport key."""
        if sport_key in self._cache:
            return self._cache[sport_key]

        data = self._get(
            f"{THEODDS_BASE}/sports/{sport_key}/odds",
            params={
                "apiKey":     self.key,
                "regions":    "uk,eu",
                "markets":    "h2h,totals",
                "oddsFormat": "decimal",
            }
        )

        records = []
        if isinstance(data, list):
            for match in data:
                try:
                    home     = match["home_team"]
                    away     = match["away_team"]
                    commence = match.get("commence_time", "")[:10]
                    parsed   = self._parse_bookmakers(match.get("bookmakers", []), home, away)
                    if parsed:
                        records.append({"home": home, "away": away,
                                        "date": commence, **parsed})
                except (KeyError, TypeError):
                    continue

        self._cache[sport_key] = records
        logger.info(f"TheOddsAPI {sport_key} → {len(records)} matches")
        return records

    def _parse_bookmakers(self, bookmakers: List[Dict],
                          home_team: str = "", away_team: str = "") -> Dict:
        """
        Parse bookmaker markets into odds dict.
        home_team / away_team are the exact strings from match["home_team"] /
        match["away_team"] — used to correctly assign 1/X/2 without relying
        on alphabetical sort (which flips home/away when away name sorts first).
        """
        preferred = ["bet365", "pinnacle", "unibet", "williamhill", "bwin"]
        bm = None
        for pref in preferred:
            bm = next((b for b in bookmakers if b.get("key") == pref), None)
            if bm:
                break
        if not bm and bookmakers:
            bm = bookmakers[0]
        if not bm:
            return {}

        result = {}
        for market in bm.get("markets", []):
            key      = market.get("key", "")
            outcomes = {o["name"]: float(o["price"])
                        for o in market.get("outcomes", [])
                        if o.get("price")}

            if key == "h2h":
                # Use exact home/away team names supplied by caller.
                # TheOddsAPI outcome names match match["home_team"] / match["away_team"].
                if home_team and home_team in outcomes:
                    result["1"] = outcomes[home_team]
                    result["X"] = outcomes.get("Draw", "-")
                    result["2"] = outcomes.get(away_team, "-") if away_team else "-"
                else:
                    # Fallback: alphabetical (may be wrong but better than nothing)
                    names = sorted(n for n in outcomes if n != "Draw")
                    if names:
                        result["1"] = outcomes.get(names[0],  "-")
                        result["X"] = outcomes.get("Draw",    "-")
                        result["2"] = outcomes.get(names[-1], "-") if len(names) > 1 else "-"
                if len(outcomes) == 2:
                    # No draw market (e.g. MLS extra-time rules) — still assign
                    non_draw = [n for n in outcomes if n != "Draw"]
                    if home_team in outcomes:
                        result["1"] = outcomes[home_team]
                        result["2"] = outcomes.get(away_team, "-")
                    elif len(non_draw) == 2:
                        result["1"] = outcomes[non_draw[0]]
                        result["2"] = outcomes[non_draw[1]]

            elif key == "totals":
                for o in market.get("outcomes", []):
                    if o.get("name") == "Over" and o.get("point") == 2.5:
                        result["over25"] = float(o["price"])
                    elif o.get("name") == "Under" and o.get("point") == 2.5:
                        result["under25"] = float(o["price"])

        return result

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Second-pass enrichment for rows still missing odds after APIF.
        Matches on ESPN league name → TheOddsAPI sport key → fuzzy team name.
        """
        if not self.enabled or df.empty:
            return df

        needs_odds = df["odds_source"] == "none"
        if not needs_odds.any():
            return df

        enriched = 0
        for league_name in df.loc[needs_odds, "league"].unique():
            sport_key = THEODDS_SPORT_KEYS.get(league_name)
            if not sport_key:
                continue

            records = self._load_league(sport_key)
            if not records:
                continue

            league_mask = needs_odds & (df["league"] == league_name)
            for idx, row in df[league_mask].iterrows():
                best       = None
                best_score = 0.0
                row_date   = str(row.get("kickoff", ""))[:10]

                for rec in records:
                    # Date filter: only match same calendar day
                    if row_date and rec.get("date") and rec["date"] != row_date:
                        continue
                    h_sim = _similarity(row.get("home", ""), rec["home"])
                    a_sim = _similarity(row.get("away", ""), rec["away"])
                    score = (h_sim + a_sim) / 2
                    if score > best_score and score >= FUZZY_THRESHOLD:
                        best_score = score
                        best = rec

                if best:
                    def _s(v):
                        return float(v) if v not in ("-", None, "") else "-"
                    df.at[idx, "odds_1"]      = _s(best.get("1"))
                    df.at[idx, "odds_X"]      = _s(best.get("X"))
                    df.at[idx, "odds_2"]      = _s(best.get("2"))
                    df.at[idx, "odds_over25"] = _s(best.get("over25"))
                    df.at[idx, "odds_u25"]    = _s(best.get("under25"))
                    df.at[idx, "odds_source"] = "TheOddsAPI"
                    needs_odds.at[idx] = False
                    enriched += 1
                    logger.info(f"TheOddsAPI matched: {row.get('home')} vs "
                                f"{row.get('away')} ({league_name}) score={best_score:.3f}")

        if enriched:
            logger.info(f"TheOddsAPI fallback: {enriched} additional fixtures enriched")
        return df

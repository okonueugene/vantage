"""
run_predictions.py
──────────────────
Vantage Engine v4.1 — Full Production Runner

Full pipeline every run:
  1. Check store for today's cached fixtures → skip API if fresh
  2. Fetch unplayed fixtures (ESPN API, free, no key)
  3. Persist fixtures + odds snapshots to SQLite store
  4. Load team form from cache (TTL 24h) or fetch + cache
  5. Classify regime from stored league stats (real data) or prior fallback
  6. Compute DRS from stored referee stats or default
  7. Run full Bayesian predictor (blend weight from stored Brier scores)
  8. Apply full risk engine (meta-error + vol regime + drawdown governor)
  9. Build dual portfolio (2 singles + 4-leg parlay)
 10. Save predictions to store + write JSON output

Output files:
  predictions_YYYYMMDD_HHMM.json   - timestamped archive
  predictions_latest.json           - always current

Usage:
  python run_predictions.py
  python run_predictions.py --bankroll 15000
  python run_predictions.py --min-ev 4.0
  python run_predictions.py settle CANONICAL_ID MARKET OUTCOME CLOSING_ODDS PLACED_ODDS STAKE PNL
  python run_predictions.py stats
"""

import argparse
import json
import logging
import math
import random
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from store import (
    init_db, upsert_fixture, get_todays_fixtures,
    save_odds_snapshot, get_line_movement,
    save_prediction, save_portfolio_predictions,
    classify_regime_from_store, get_cached_team_form,
    get_ref_red_card_rate, get_clv_summary, get_brier_scores,
    get_brier_by_market, get_hit_rate_by_market, storage_summary, settle_result,
    stale_fixture_guard, _conn, save_match_result_for_stats,
    mark_fixture_finished, backfill_brier_from_league_stats,
)
from predictor import Predictor, LEAGUE_PRIORS, EV_THRESHOLD
from portfolio import PortfolioBuilder, BANKROLL_DEFAULT
from risk import RiskEngine, ErrorSignal
from normalizer import normalize_team
from calibration import CalibrationLab

# ── Windows-safe logging setup ────────────────────────────────────────────────
# Force stdout to UTF-8 on Windows (CP1252 can't encode emoji).
# If that fails (e.g. redirected pipe), strip non-ASCII chars from log records.
import io as _io

def _safe_stdout_stream():
    """Return a UTF-8 stdout stream that works on Windows CP1252 consoles."""
    try:
        return _io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except AttributeError:
        # sys.stdout.buffer not available (e.g. IDLE, some IDEs)
        return sys.stdout


class _AsciiSafeFilter(logging.Filter):
    """Strip characters outside Basic Multilingual Plane from log messages."""
    def filter(self, record):
        if isinstance(record.msg, str):
            record.msg = record.msg.encode("ascii", "replace").decode("ascii")
        return True


_stdout_stream = _safe_stdout_stream()
_console_handler = logging.StreamHandler(_stdout_stream)
_console_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-5s | %(name)-20s | %(message)s")
)

# Only add ASCII filter if we couldn't get a real UTF-8 stream
if _stdout_stream is sys.stdout:
    _console_handler.addFilter(_AsciiSafeFilter())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)-20s | %(message)s",
    handlers=[
        logging.FileHandler("vantage.log", encoding="utf-8"),
        _console_handler,
    ]
)
logger = logging.getLogger("Vantage.Main")

ESPN_LEAGUES = {
    # ── England ───────────────────────────────────────────────────────────────
    "eng.1":          "Premier League",
    "eng.2":          "Championship",
    "eng.3":          "League One",
    "eng.4":          "League Two",
    # ── Spain ─────────────────────────────────────────────────────────────────
    "esp.1":          "La Liga",
    "esp.2":          "Segunda Division",
    # ── Germany ───────────────────────────────────────────────────────────────
    "ger.1":          "Bundesliga",
    "ger.2":          "2. Bundesliga",
    # ── Italy ─────────────────────────────────────────────────────────────────
    "ita.1":          "Serie A",
    "ita.2":          "Serie B",
    # ── France ────────────────────────────────────────────────────────────────
    "fra.1":          "Ligue 1",
    "fra.2":          "Ligue 2",
    # ── Netherlands ───────────────────────────────────────────────────────────
    "ned.1":          "Eredivisie",
    # ── Portugal ──────────────────────────────────────────────────────────────
    "por.1":          "Primeira Liga",
    # ── Turkey ────────────────────────────────────────────────────────────────
    "tur.1":          "Süper Lig",
    # ── Scotland ──────────────────────────────────────────────────────────────
    "sco.1":          "Scottish Premiership",
    "sco.2":          "Scottish Championship",
    # ── Belgium ───────────────────────────────────────────────────────────────
    "bel.1":          "Belgian Pro League",
    # ── Other Europe ──────────────────────────────────────────────────────────
    "gre.1":          "Super League Greece",
    "den.1":          "Danish Superliga",
    "swe.1":          "Allsvenskan",
    "nor.1":          "Eliteserien",
    "sui.1":          "Swiss Super League",
    "aut.1":          "Austrian Bundesliga",
    "cze.1":          "Czech First League",
    # pol.1 (Ekstraklasa) — returns HTTP 400, ESPN geo-restricted. Priors available for manual odds.
    # ukr.1 (Ukrainian Premier League) — ESPN suspended coverage. Priors available for manual odds.
    "rus.1":          "Russian Premier League",
    # ── Americas ──────────────────────────────────────────────────────────────
    "usa.1":          "MLS",
    "mex.1":          "Liga MX",
    "bra.1":          "Brasileirao",
    "arg.1":          "Argentine Primera",
    # ── Asia / Oceania ────────────────────────────────────────────────────────
    "ksa.1":          "Saudi Pro League",
    "aus.1":          "A-League",
    "jpn.1":          "J1 League",
    "chn.1":          "Chinese Super League",
    # ── UEFA club competitions ────────────────────────────────────────────────
    "uefa.champions": "Champions League",
    "uefa.europa":    "Europa League",
    "uefa.europa.conf": "Conference League",
    # ── Domestic cups ─────────────────────────────────────────────────────────
    "eng.fa":             "FA Cup",
    "eng.league_cup":     "EFL Cup",
    "esp.copa_del_rey":   "Copa del Rey",
    "ger.dfb_pokal":      "DFB-Pokal",
    "ita.coppa_italia":   "Coppa Italia",
    "fra.coupe_de_france":"Coupe de France",
    "ned.knvb":           "KNVB Cup",
    "por.taca":           "Taça de Portugal",
    "sco.fa":             "Scottish FA Cup",
    "sco.league_cup":     "Scottish League Cup",
    "tur.cup":            "Turkish Cup",
    "bel.cup":            "Belgian Cup",
    "gre.cup":            "Greek Cup",
    "bra.copa":           "Copa do Brasil",
    "arg.copa":           "Copa Argentina",
    "usa.open":           "US Open Cup",
    # ── International ─────────────────────────────────────────────────────────
    # World Cup Qualifiers (by confederation)
    "fifa.worldq.uefa":      "UEFA World Cup Qualifiers",
    "fifa.worldq.conmebol":  "CONMEBOL World Cup Qualifiers",
    "fifa.worldq.concacaf":  "CONCACAF World Cup Qualifiers",
    "fifa.worldq.caf":       "CAF World Cup Qualifiers",
    "fifa.worldq.afc":       "AFC World Cup Qualifiers",
    "fifa.worldq.ofc":       "OFC World Cup Qualifiers",
    # Continental tournaments
    "uefa.nations":          "UEFA Nations League",
    "concacaf.nations":      "CONCACAF Nations League",
    "conmebol.america":      "Copa America",
    "caf.nations":           "AFCON",
    "afc.cup":               "AFC Asian Cup",
    # Friendlies (lower weight — treated as prior-only)
    "fifa.friendly":         "International Friendlies",
}
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
TOP_LEAGUES = {
    # England
    "Premier League", "Championship", "League One", "League Two",
    # Big 4 Europe
    "La Liga", "Bundesliga", "Serie A", "Ligue 1",
    # Other Europe tier 1
    "Eredivisie", "Primeira Liga", "Süper Lig", "Scottish Premiership",
    "Belgian Pro League",
    # Europe tier 2 / domestic seconds
    "Segunda Division", "2. Bundesliga", "Serie B", "Ligue 2",
    "Scottish Championship", "Super League Greece", "Danish Superliga",
    "Allsvenskan", "Eliteserien", "Swiss Super League", "Austrian Bundesliga",
    "Czech First League", "Ekstraklasa", "Ukrainian Premier League",
    "Russian Premier League",
    # Americas
    "MLS", "Liga MX", "Brasileirao", "Argentine Primera",
    # Asia / Middle East / Oceania
    "Saudi Pro League", "A-League", "J1 League", "Chinese Super League",
    # UEFA competitions
    "Champions League", "Europa League", "Conference League",
    # International
    "UEFA World Cup Qualifiers", "CONMEBOL World Cup Qualifiers",
    "CONCACAF World Cup Qualifiers", "CAF World Cup Qualifiers",
    "AFC World Cup Qualifiers", "OFC World Cup Qualifiers",
    "UEFA Nations League", "CONCACAF Nations League",
    "Copa America", "AFCON", "AFC Asian Cup",
    "International Friendlies",
}

# Canonical league name mapping — handles ESPN returning variant spellings
_LEAGUE_ALIASES = {
    "Super Lig":                 "Süper Lig",
    "Turkish Super Lig":         "Süper Lig",
    "Belgian First Division A":  "Belgian Pro League",
    "Jupiler Pro League":        "Belgian Pro League",
    "Australian A-League":       "A-League",
    "Major League Soccer":       "MLS",
    "Saudi Professional League": "Saudi Pro League",
    "Segunda División":          "Segunda Division",
    "Liga MX Apertura":          "Liga MX",
    "Liga MX Clausura":          "Liga MX",
    "Brasileirão":               "Brasileirao",
    "Campeonato Brasileiro":     "Brasileirao",
    "Liga Profesional":          "Argentine Primera",
    "Primera División":          "Argentine Primera",
    "English League Championship": "Championship",
    "EFL Championship":          "Championship",
    "EFL League One":            "League One",
    "EFL League Two":            "League Two",
    "UEFA Europa Conference League": "Conference League",
    "UEFA Champions League":     "Champions League",
    "UEFA Europa League":        "Europa League",
    # ── Cup aliases ───────────────────────────────────────────────────────────
    "Emirates FA Cup":           "FA Cup",
    "The FA Cup":                "FA Cup",
    "Carabao Cup":               "EFL Cup",
    "EFL Cup":                   "EFL Cup",
    "League Cup":                "EFL Cup",
    "Copa del Rey":              "Copa del Rey",
    "DFB Pokal":                 "DFB-Pokal",
    "DFB-Pokal":                 "DFB-Pokal",
    "Coppa Italia":              "Coppa Italia",
    "Coupe de France":           "Coupe de France",
    "KNVB Cup":                  "KNVB Cup",
    "KNVB Beker":                "KNVB Cup",
    "Taca de Portugal":          "Taça de Portugal",
    "Taça de Portugal":          "Taça de Portugal",
    "Scottish Cup":              "Scottish FA Cup",
    "Scottish FA Cup":           "Scottish FA Cup",
    "Scottish League Cup":       "Scottish League Cup",
    "Viaplay Cup":               "Scottish League Cup",
    "Turkish Cup":               "Turkish Cup",
    "Ziraat Turkish Cup":        "Turkish Cup",
    "Belgian Cup":               "Belgian Cup",
    "Croky Cup":                 "Belgian Cup",
    "Greek Cup":                 "Greek Cup",
    "Copa do Brasil":            "Copa do Brasil",
    "Copa Argentina":            "Copa Argentina",
    "U.S. Open Cup":             "US Open Cup",
    "Lamar Hunt U.S. Open Cup":  "US Open Cup",
    # International — ESPN uses various naming conventions
    "FIFA World Cup Qualifying - UEFA":     "UEFA World Cup Qualifiers",
    "FIFA World Cup Qualifying - CONMEBOL": "CONMEBOL World Cup Qualifiers",
    "FIFA World Cup Qualifying - CONCACAF": "CONCACAF World Cup Qualifiers",
    "FIFA World Cup Qualifying - CAF":      "CAF World Cup Qualifiers",
    "FIFA World Cup Qualifying - AFC":      "AFC World Cup Qualifiers",
    "FIFA World Cup Qualifying - OFC":      "OFC World Cup Qualifiers",
    "UEFA Nations League":                  "UEFA Nations League",
    "CONCACAF Nations League":              "CONCACAF Nations League",
    "Copa America":                         "Copa America",
    "Africa Cup of Nations":                "AFCON",
    "AFC Asian Cup":                        "AFC Asian Cup",
    "International Friendly":               "International Friendlies",
    "International Friendlies":             "International Friendlies",
    "Friendlies":                           "International Friendlies",
}

def _normalize_league(name: str) -> str:
    """Resolve ESPN league name variants to our canonical names."""
    return _LEAGUE_ALIASES.get(name, name)


# ══════════════════════════════════════════════
# STEP 1 — Fetch or load from store
# ══════════════════════════════════════════════
def _patch_odds_fetcher(fixtures: List[Dict], match_date: str) -> List[Dict]:
    """
    Enrich fixture odds via OddsFetcher.enrich_with_odds(df).

    Strategy (2026-03-08 rewrite):
      - API-Football /odds endpoint has NO team names.
      - OddsFetcher fetches /fixtures?date= (has names) + /odds?date= (has odds),
        joins on fixture.id, then fuzzy-matches to our ESPN rows.
      - enrich_with_odds() accepts a DataFrame and returns it enriched.
    """
    if not fixtures:
        return fixtures

    # ── Pass 1: Betika (primary) ──────────────────────────────────────────────
    # Betika is the most reliable local odds source. Runs first so APIF/TheOddsAPI
    # only fill markets Betika couldn't cover (mainly over25/under25 goals lines,
    # which Betika's API corrupts for lines 0.5–3.5).
    # Writes directly to fx["odds"] in-place.
    try:
        from betika_odds_fetcher import BetikaOddsFetcher
        betika = BetikaOddsFetcher()
        n_betika = betika.enrich(fixtures)
        if n_betika:
            logger.info(f"Betika: {n_betika}/{len(fixtures)} fixtures enriched (1X2/BTTS/corners/cards)")
        else:
            logger.info("Betika: 0 fixtures matched — check team names or kickoff times")
    except Exception as e:
        logger.warning(f"Betika pass failed: {e}")

    # ── Pass 2: API-Football (fill gaps) ─────────────────────────────────────
    # APIF fills markets not covered by Betika — primarily over25/under25.
    # Does NOT overwrite markets already set by Betika.
    try:
        import pandas as pd
        from odds_fetcher import OddsFetcher
        fetcher = OddsFetcher()

        if not fetcher.enabled:
            logger.debug("OddsFetcher: no API_FOOTBALL_KEY — skipping")
        else:
            logger.info("OddsFetcher: enriching odds via date-based bulk fetch…")

            rows = []
            for i, fx in enumerate(fixtures):
                rows.append({
                    "_idx":    i,
                    "home":    fx.get("home", ""),
                    "away":    fx.get("away", ""),
                    "league":  fx.get("league", ""),
                    "kickoff": fx.get("kickoff") or match_date,
                })
            df = pd.DataFrame(rows)
            df = fetcher.enrich_with_odds(df)

            # Map APIF odds into fx["odds"] — only fill missing markets
            apif_map = {
                "home_win": "odds_1", "draw": "odds_X", "away_win": "odds_2",
                "over25": "odds_over25", "under25": "odds_u25",
                "btts_yes": "odds_btts_yes",
                "corners_over": "odds_corners_over", "cards_over": "odds_cards_over",
            }
            apif_enriched = 0
            for _, row in df.iterrows():
                if row.get("odds_source") in (None, "", "none"):
                    continue
                fx = fixtures[int(row["_idx"])]
                fx_odds = fx.setdefault("odds", {})
                added = False
                for market, col in apif_map.items():
                    val = row.get(col)
                    if val and val != "-" and market not in fx_odds:
                        try:
                            fx_odds[market] = float(val)
                            cid = fx.get("canonical_id", "")
                            if cid:
                                save_odds_snapshot(cid, market, float(val),
                                                   source=row["odds_source"])
                            added = True
                        except (TypeError, ValueError):
                            pass
                if added:
                    apif_enriched += 1
                    fx["_odds_fresh"] = True
            logger.info(f"API-Football: gap-filled {apif_enriched} fixtures")

    except ImportError:
        logger.debug("odds_fetcher.py not found — skipping")
    except Exception as e:
        logger.warning(f"OddsFetcher failed (non-fatal): {e}")

    # ── Pass 3: TheOddsAPI (fill remaining gaps) ──────────────────────────────
    # Last resort — mainly useful for over25/under25 when APIF also missed.
    # Does NOT overwrite markets already set by Betika or APIF.
    try:
        import pandas as pd
        from odds_fetcher import TheOddsAPIFetcher
        ta_fetcher = TheOddsAPIFetcher()
        if ta_fetcher.enabled:
            rows = []
            for i, fx in enumerate(fixtures):
                rows.append({
                    "_idx":    i,
                    "home":    fx.get("home", ""),
                    "away":    fx.get("away", ""),
                    "league":  fx.get("league", ""),
                    "kickoff": fx.get("kickoff") or match_date,
                })
            df_ta = pd.DataFrame(rows)
            df_ta = ta_fetcher.enrich(df_ta)

            ta_map = {
                "home_win": "odds_1", "draw": "odds_X", "away_win": "odds_2",
                "over25": "odds_over25", "under25": "odds_u25",
            }
            ta_enriched = 0
            for _, row in df_ta.iterrows():
                if row.get("odds_source") in (None, "", "none"):
                    continue
                fx = fixtures[int(row["_idx"])]
                fx_odds = fx.setdefault("odds", {})
                added = False
                for market, col in ta_map.items():
                    val = row.get(col)
                    if val and val != "-" and market not in fx_odds:
                        try:
                            fx_odds[market] = float(val)
                            cid = fx.get("canonical_id", "")
                            if cid:
                                save_odds_snapshot(cid, market, float(val),
                                                   source="TheOddsAPI")
                            added = True
                        except (TypeError, ValueError):
                            pass
                if added:
                    ta_enriched += 1
                    fx["_odds_fresh"] = True
            logger.info(f"TheOddsAPI: gap-filled {ta_enriched} fixtures")
    except Exception as e:
        logger.debug(f"TheOddsAPI pass skipped: {e}")

    # ── Persist Betika-specific markets (corners_*, cards_* granular lines) ───
    for fx in fixtures:
        fx_odds = fx.get("odds", {})
        betika_only = {k: v for k, v in fx_odds.items()
                       if k.startswith(("corners_", "cards_", "btts_"))
                       and isinstance(v, (int, float))}
        cid = fx.get("canonical_id", "")
        if cid and betika_only:
            for market, val in betika_only.items():
                try:
                    save_odds_snapshot(cid, market, float(val), source="Betika")
                except Exception:
                    pass

    enriched = sum(1 for fx in fixtures if fx.get("odds") and fx.get("_odds_fresh"))
    total_with_any = sum(1 for fx in fixtures if fx.get("odds"))
    logger.info(f"OddsFetcher: {enriched}/{len(fixtures)} fixtures freshly enriched this session ({total_with_any} have any odds)")
    return fixtures


def backfill_results_from_espn(days_back: int = 7) -> int:
    """
    Fetch completed match scores from ESPN for the last N days → league_stats.

    Speed optimisation: one HTTP call per league (ESPN accepts comma-separated
    dates OR a range via the dates param). We use a date-range string covering
    all days_back days in a single request per league (~15 calls total, ~10s).

    Idempotent: league_stats uses INSERT OR IGNORE so reruns are safe.
    Does NOT touch the results table (placed-bets only).
    """
    from datetime import timedelta
    now    = datetime.now(timezone.utc)
    season = "2025-26"
    new_count = 0

    # Build comma-separated date string: ESPN accepts ?dates=20260301,20260302,...
    dates_param = ",".join(
        (now - timedelta(days=d)).strftime("%Y%m%d")
        for d in range(1, days_back + 1)
    )

    # Check which dates we already have data for — skip those leagues entirely
    with _conn() as con:
        existing_dates = {
            r["match_date"] for r in
            con.execute("SELECT DISTINCT match_date FROM league_stats").fetchall()
        }

    need_dates = set(
        (now - timedelta(days=d)).strftime("%Y-%m-%d")
        for d in range(1, days_back + 1)
    ) - existing_dates

    if not need_dates:
        logger.info("Backfill: all dates already in league_stats — skipping")
        return 0

    logger.info(f"Backfill: fetching {len(need_dates)} missing date(s) across {len(ESPN_LEAGUES)} leagues")

    # Fetch per-date (not comma-separated) — ESPN multi-date param is unreliable
    # across leagues. Sequential but fast (0.2–0.4s sleep = ~30s for 7 days × 53 leagues).
    for days_ago in range(1, days_back + 1):
        target     = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        espn_date  = target.replace("-", "")

        if target not in need_dates:
            continue

        for code, league_name in ESPN_LEAGUES.items():
            url = ESPN_BASE.format(league=code) + f"?dates={espn_date}"
            try:
                time.sleep(random.uniform(0.2, 0.4))
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (VantageEngine/4.1)",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode())
            except Exception as e:
                logger.debug(f"Backfill ESPN {league_name}: {e}")
                continue

            league_norm = _normalize_league(league_name)

            for event in data.get("events", []):
                try:
                    comp   = event["competitions"][0]
                    status = comp.get("status", {}).get("type", {})
    
                    if status.get("state") != "post" or not status.get("completed", False):
                        continue
    
                    competitors = comp.get("competitors", [])
                    if len(competitors) < 2:
                        continue
    
                    home_obj = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                    away_obj = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
    
                    home = normalize_team(home_obj["team"]["displayName"])
                    away = normalize_team(away_obj["team"]["displayName"])
    
                    home_goals = int(home_obj.get("score", 0) or 0)
                    away_goals = int(away_obj.get("score", 0) or 0)
    
                    match_date = event.get("date", "")[:10]  # "2026-03-07T..."[:10]
                    if match_date not in need_dates:
                        continue  # already have this date
    
                    cid = (f"{re.sub(r'[^A-Za-z]','',home)[:4].upper()}_"
                           f"{re.sub(r'[^A-Za-z]','',away)[:4].upper()}_"
                           f"{match_date.replace('-','')}")
    
                    upsert_fixture({
                        "canonical_id": cid,
                        "home":         home,
                        "away":         away,
                        "league":       league_norm,
                        "kickoff_utc":  match_date + " 00:00 UTC",
                        "match_date":   match_date,
                        "status":       "finished",
                        "source":       "ESPN_BACKFILL",
                        "odds":         {},
                    })
    
                    save_match_result_for_stats(
                        league=league_norm,
                        season=season,
                        match_date=match_date,
                        home=home,
                        away=away,
                        home_goals=home_goals,
                        away_goals=away_goals,
                    )

                    # Extract post-match stats (corners, shots, cards, possession)
                    # ESPN puts these in comp["statistics"] for completed matches
                    try:
                        from store import save_match_stats
                        stats_raw = comp.get("statistics", [])
                        def _stat(name):
                            for s in stats_raw:
                                if s.get("name","").lower() == name.lower():
                                    vals = s.get("stats", s.get("values", []))
                                    if len(vals) >= 2:
                                        try: return int(float(vals[0])), int(float(vals[1]))
                                        except: pass
                            return None, None

                        ch, ca = _stat("cornerKicks")
                        sh, sa = _stat("totalShots")
                        soth, sota = _stat("shotsOnTarget")
                        yh, ya = _stat("yellowCards")
                        rh, ra = _stat("redCards")
                        ph, pa = _stat("possessionPct")

                        if any(v is not None for v in [ch, sh, yh]):
                            save_match_stats(cid, {
                                "xg_home":         None,
                                "xg_away":         None,
                                "corners_home":    ch,
                                "corners_away":    ca,
                                "shots_home":      sh,
                                "shots_away":      sa,
                                "shots_on_home":   soth,
                                "shots_on_away":   sota,
                                "cards_home":      (yh or 0) + (rh or 0),
                                "cards_away":      (ya or 0) + (ra or 0),
                                "possession_home": ph,
                                "possession_away": pa,
                                "source":          "ESPN_BACKFILL",
                            })
                    except Exception:
                        pass

                    new_count += 1
    
                except Exception as e:
                    logger.debug(f"Backfill parse error {league_name}: {e}")
                    continue
    
    logger.info(f"Backfill complete — {new_count} new results written to league_stats")
    return new_count


def fetch_or_load_fixtures(
    target_date: Optional[str] = None,
    lookahead: int = 2,
) -> Tuple[List[Dict], bool]:
    """
    Load fixtures for target_date (default: today UTC).
    If no fixtures found, look ahead up to `lookahead` days.
    Always runs stale_fixture_guard to drop matches that have already kicked off.

    Cache-hit logic: if fixtures are cached but none of the newly-added
    international leagues are represented, we do a targeted re-fetch for
    those leagues and merge the results.
    """
    today = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Try cache first for the target date
    ALL_MARKETS = ["home_win","draw","away_win","over25","under25","corners_over","cards_over"]

    # International leagues that may not be in older cache entries
    INTL_LEAGUES = {
        "UEFA World Cup Qualifiers", "CONMEBOL World Cup Qualifiers",
        "CONCACAF World Cup Qualifiers", "CAF World Cup Qualifiers",
        "AFC World Cup Qualifiers", "OFC World Cup Qualifiers",
        "UEFA Nations League", "CONCACAF Nations League",
        "Copa America", "AFCON", "AFC Asian Cup",
        "International Friendlies",
    }
    INTL_ESPN_CODES = {
        "UEFA World Cup Qualifiers":     "fifa.worldq.uefa",
        "CONMEBOL World Cup Qualifiers": "fifa.worldq.conmebol",
        "CONCACAF World Cup Qualifiers": "fifa.worldq.concacaf",
        "CAF World Cup Qualifiers":      "fifa.worldq.caf",
        "AFC World Cup Qualifiers":      "fifa.worldq.afc",
        "OFC World Cup Qualifiers":      "fifa.worldq.ofc",
        "UEFA Nations League":           "uefa.nations",
        "CONCACAF Nations League":       "concacaf.nations",
        "Copa America":                  "conmebol.america",
        "AFCON":                         "caf.nations",
        "AFC Asian Cup":                 "afc.cup",
        "International Friendlies":      "fifa.friendly",
    }

    cached = get_todays_fixtures(today)
    if cached:
        logger.info(f"Store hit: {len(cached)} fixtures cached for {today}")
        for fx in cached:
            if "odds" not in fx or not fx["odds"]:
                fx["odds"] = {}
            for m in ALL_MARKETS:
                if m not in fx["odds"]:
                    v = _get_stored_odds(fx.get("canonical_id",""), m)
                    if v:
                        fx["odds"][m] = v

        # Check if any international fixtures are missing from cache
        cached_leagues = {fx.get("league","") for fx in cached}
        missing_intl = INTL_LEAGUES - cached_leagues
        if missing_intl:
            logger.info(f"Cache hit but {len(missing_intl)} international league(s) missing — fetching: {', '.join(sorted(missing_intl))}")
            date_fmt = today.replace("-","")
            import urllib.request as _ur, json as _json
            ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{}/scoreboard"
            new_fixtures = []
            for league_name in missing_intl:
                code = INTL_ESPN_CODES.get(league_name)
                if not code:
                    continue
                try:
                    url = ESPN_SCOREBOARD.format(code) + f"?dates={date_fmt}"
                    req = _ur.Request(url, headers={"User-Agent": "VantageEngine/4.1"})
                    with _ur.urlopen(req, timeout=8) as r:
                        data = _json.loads(r.read())
                    events = data.get("events", [])
                    if not events:
                        continue
                    logger.info(f"  ✓ {league_name}: {len(events)} fixture(s)")
                    for event in events:
                        comps = event.get("competitions", [{}])[0]
                        competitors = comps.get("competitors", [{},{}])
                        home = competitors[0].get("team", {}).get("displayName", "")
                        away = competitors[1].get("team", {}).get("displayName", "")
                        if not home or not away:
                            continue
                        kick_str = event.get("date", "")
                        fx = {
                            "home": home, "away": away,
                            "league": league_name,
                            "kickoff": kick_str,
                            "espn_id": event.get("id",""),
                            "odds": {},
                            "canonical_id": f"{home[:4].upper()}_{away[:4].upper()}_{today.replace('-','')}",
                        }
                        upsert_fixture(fx)
                        new_fixtures.append(fx)
                except Exception as e:
                    logger.debug(f"  International fetch failed {league_name}: {e}")
            if new_fixtures:
                logger.info(f"  Added {len(new_fixtures)} international fixture(s) to cache")
                cached = cached + new_fixtures

        fixtures, stale = stale_fixture_guard(cached)
        _log_stale(stale)
        fixtures = _patch_odds_fetcher(fixtures, today)
        return fixtures, True

    # Fetch from ESPN — try target date first, then look ahead
    logger.info("Store miss — fetching from ESPN API")
    from datetime import timedelta as _td
    dates_to_try = [today]
    base = datetime.strptime(today, "%Y-%m-%d")
    for d in range(1, lookahead + 1):
        dates_to_try.append((base + _td(days=d)).strftime("%Y-%m-%d"))

    for attempt_date in dates_to_try:
        raw = _fetch_espn(attempt_date)
        if raw:
            if attempt_date != today:
                logger.info(
                    f"No games today ({today}) — showing fixtures for {attempt_date} "
                    f"(use --date {attempt_date} to target this day explicitly)"
                )
            fixtures, stale = stale_fixture_guard(raw)
            _log_stale(stale)
            fixtures = _patch_odds_fetcher(fixtures, attempt_date)
            return fixtures, False

    logger.info(
        f"No unplayed fixtures found for {today}"
        + (f" or next {lookahead} days" if lookahead else "")
        + ". Try --date YYYY-MM-DD or --lookahead N."
    )
    return [], False


def _log_stale(stale: list) -> None:
    """Log dropped fixtures with clear reason."""
    for cid, kickoff_str, mins_until in stale:
        if mins_until >= 0:
            logger.info(f"Skipping {cid} — kicks off in {mins_until:.0f} min (< 5 min cutoff)")
        elif mins_until > -105:
            logger.info(f"Skipping {cid} — kicked off {abs(mins_until):.0f} min ago (live, marked)")
        else:
            logger.info(f"Skipping {cid} — match finished ({kickoff_str}), marked in store")


def _get_stored_odds(canonical_id: str, market: str) -> Optional[float]:
    from store import _conn
    with _conn() as con:
        r = con.execute(
            "SELECT odds_decimal FROM odds WHERE canonical_id=? AND market=? ORDER BY snapshot_time DESC LIMIT 1",
            (canonical_id, market)
        ).fetchone()
    return r["odds_decimal"] if r else None


def _fetch_espn(target_date: str) -> List[Dict]:
    """
    Fetch unplayed fixtures from ESPN API for target_date (YYYY-MM-DD).
    ESPN scoreboard accepts a `dates` param (YYYYMMDD) to get a specific day.
    """
    fixtures = []
    seen     = set()
    espn_date = target_date.replace("-", "")  # 20260302

    for code, league_name in ESPN_LEAGUES.items():
        url = ESPN_BASE.format(league=code) + f"?dates={espn_date}"
        try:
            time.sleep(random.uniform(0.4, 0.9))
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (VantageEngine/4.1)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            logger.warning(f"ESPN {league_name}: {e}")
            continue

        for event in data.get("events", []):
            try:
                comp   = event["competitions"][0]
                status = comp.get("status", {}).get("type", {})
                if status.get("state") != "pre" or status.get("completed", False):
                    continue

                competitors = comp.get("competitors", [])
                if len(competitors) < 2:
                    continue

                home_obj = next((c for c in competitors if c.get("homeAway")=="home"), competitors[0])
                away_obj = next((c for c in competitors if c.get("homeAway")=="away"), competitors[1])
                home = normalize_team(home_obj["team"]["displayName"])
                away = normalize_team(away_obj["team"]["displayName"])

                key = f"{home}_{away}"
                if key in seen:
                    continue
                seen.add(key)

                kickoff_raw = event.get("date","")
                kickoff_str = _fmt_kickoff(kickoff_raw)
                cid = (f"{re.sub(r'[^A-Za-z]','',home)[:4].upper()}_"
                       f"{re.sub(r'[^A-Za-z]','',away)[:4].upper()}_"
                       f"{target_date.replace('-','')}")

                odds = _extract_espn_odds(comp)

                fixture = {
                    "canonical_id": cid,
                    "home":         home,
                    "away":         away,
                    "league":       _normalize_league(league_name),
                    "kickoff_utc":  kickoff_str,
                    "match_date":   target_date,
                    "status":       "unplayed",
                    "source":       "ESPN_API",
                    "odds":         odds,
                }

                upsert_fixture(fixture)
                for market, val in odds.items():
                    if isinstance(val, float) and val > 1.01:
                        save_odds_snapshot(cid, market, val, source="ESPN_API")

                fixtures.append(fixture)
            except (KeyError, IndexError):
                continue

        n = sum(1 for f in fixtures if f["league"] == league_name)
        logger.info(f"ESPN {league_name}: {n} unplayed")

    logger.info(f"Fetched + stored {len(fixtures)} fixtures for {target_date}")
    return fixtures


def _fmt_kickoff(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso.replace("Z","+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso


def _ml_to_decimal(ml) -> float:
    try:
        ml = int(ml)
        return round(ml/100+1,3) if ml>0 else round(100/abs(ml)+1,3)
    except Exception:
        return 0.0


def _extract_espn_odds(comp: Dict) -> Dict:
    odds = {}
    try:
        od = comp.get("odds",[{}])
        if isinstance(od, list) and od:
            o = od[0]
            hl = o.get("homeTeamOdds",{}).get("moneyLine")
            al = o.get("awayTeamOdds",{}).get("moneyLine")
            dl = o.get("drawOdds",    {}).get("moneyLine")
            if hl: odds["home_win"] = _ml_to_decimal(hl)
            if al: odds["away_win"] = _ml_to_decimal(al)
            if dl: odds["draw"]     = _ml_to_decimal(dl)
    except Exception:
        pass
    return odds


# ══════════════════════════════════════════════
# STEP 2 — Enrich from store
# ══════════════════════════════════════════════
def enrich_fixture(fixture: Dict) -> Dict:
    league = fixture["league"]
    home   = fixture["home"]
    away   = fixture["away"]
    cid    = fixture.get("canonical_id","")

    regime    = classify_regime_from_store(league)
    home_form = get_cached_team_form(home, league) or {}
    away_form = get_cached_team_form(away, league) or {}

    # For cup/European competitions, teams are stored under their domestic league.
    # Fall back to any-league lookup if the cup-specific lookup returned nothing.
    CUP_LEAGUES = {
        "Champions League", "Europa League", "Conference League",
        "FA Cup", "EFL Cup", "Coppa Italia", "Copa del Rey", "DFB-Pokal",
        "Coupe de France", "Copa Argentina", "Copa do Brasil", "US Open Cup",
        # International — use domestic club xG as fallback for national team form
        "UEFA World Cup Qualifiers", "CONMEBOL World Cup Qualifiers",
        "CONCACAF World Cup Qualifiers", "CAF World Cup Qualifiers",
        "AFC World Cup Qualifiers", "OFC World Cup Qualifiers",
        "UEFA Nations League", "CONCACAF Nations League",
        "Copa America", "AFCON", "AFC Asian Cup",
        "International Friendlies",
    }
    if league in CUP_LEAGUES:
        def _has_useful_form(f):
            return f and (f.get("avg_xg_for") is not None or f.get("avg_goals_for") is not None)

        if not _has_useful_form(home_form):
            with _conn() as con:
                rows = con.execute("""
                    SELECT * FROM team_form WHERE team=?
                    AND avg_xg_for IS NOT NULL
                    ORDER BY fetched_at DESC LIMIT 1
                """, (home,)).fetchone()
                if rows:
                    rest = home_form.get("days_rest")  # preserve cup rest days
                    home_form = dict(rows)
                    if rest is not None:
                        home_form["days_rest"] = rest  # cup rest days are more accurate

        if not _has_useful_form(away_form):
            with _conn() as con:
                rows = con.execute("""
                    SELECT * FROM team_form WHERE team=?
                    AND avg_xg_for IS NOT NULL
                    ORDER BY fetched_at DESC LIMIT 1
                """, (away,)).fetchone()
                if rows:
                    rest = away_form.get("days_rest")
                    away_form = dict(rows)
                    if rest is not None:
                        away_form["days_rest"] = rest

    # Only use real form data — do NOT fall back to league prior as fake xG.
    # If no form data exists, leave xG as None so predictor knows it has no team data.
    home_xg     = home_form.get("avg_xg_for")   # None if no cached form
    away_xg     = away_form.get("avg_xg_for")   # None if no cached form
    home_xg_src = home_form.get("xg_source") or None
    away_xg_src = away_form.get("xg_source") or None

    # Source tiers — determines 1X2 reliability in predictor:
    #   understat  → real xG (shot-quality model) — full blend, 1X2 allowed
    #   espn       → shots×0.32 proxy — reduced blend, 1X2 allowed (cautious)
    #   sofascore  → shots-based xG for most leagues — reduced blend, 1X2 allowed
    #   goals_proxy→ raw goals/g used as xG (no shot data) — 1X2 KILLED
    #   prior      → no form data at all — 1X2 KILLED
    UNDERSTAT_LEAGUES = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
    # Leagues where Sofascore stores real xG (shot-quality, not goals/g)
    SOFASCORE_REAL_XG = {
        "Eredivisie", "Belgian Pro League", "Primeira Liga", "Süper Lig",
        "Scottish Premiership", "Championship", "J1 League", "Saudi Pro League",
        "A-League", "Chinese Super League",
    }

    def _infer_src(form: dict, lg: str) -> str:
        xg_val  = form.get("avg_xg_for")
        g_val   = form.get("avg_goals_for")
        stored  = form.get("xg_source")
        if stored:
            return stored
        if xg_val:
            if lg in UNDERSTAT_LEAGUES:
                return "understat"
            if lg in SOFASCORE_REAL_XG:
                return "sofascore"
            # Sofascore ran but may have stored goals/g — tag conservatively
            return "goals_proxy"
        if g_val:
            return "goals_proxy"
        return "prior"

    home_xg_src = _infer_src(home_form, league)
    away_xg_src = _infer_src(away_form, league)

    # Fallback: goals/g as xG value — tag as goals_proxy
    if home_xg is None and home_form.get("avg_goals_for"):
        home_xg = home_form["avg_goals_for"]
        home_xg_src = "goals_proxy"
    if away_xg is None and away_form.get("avg_goals_for"):
        away_xg = away_form["avg_goals_for"]
        away_xg_src = "goals_proxy"

    ref_name = fixture.get("referee","")
    drs_base = get_ref_red_card_rate(ref_name, league) if ref_name else 0.12
    inj_bump = ((home_form.get("injury_count",0) or 0) + (away_form.get("injury_count",0) or 0)) * 0.005
    drs      = round(min(drs_base + inj_bump, 0.40), 3)

    home_pos = home_form.get("league_position") or home_form.get("position") or 10
    away_pos = away_form.get("league_position") or away_form.get("position") or 10
    motiv    = min(max(away_pos - home_pos, -10), 10)

    odds = dict(fixture.get("odds") or {})

    # Fallback: parse odds from raw_json if the odds dict is missing market prices
    # (ESPN API stores odds inside raw_json but fixture dict may not have them unpacked)
    if not any(k in odds for k in ("home_win", "draw", "away_win", "over25", "under25")):
        import json as _json
        raw = fixture.get("raw_json") or ""
        if raw:
            try:
                raw_odds = _json.loads(raw).get("odds", {})
                for market in ("home_win", "draw", "away_win", "over25", "under25",
                               "corners_over", "cards_over", "btts_yes"):
                    if market in raw_odds and market not in odds:
                        odds[market] = raw_odds[market]
            except Exception:
                pass
    for market in ["home_win","draw","away_win","over25","under25","corners_over","cards_over"]:
        drift = get_line_movement(cid, market)
        if drift is not None:
            odds[f"{market}_drift_pct"] = drift

    home_rest    = home_form.get("days_rest", 4) or 4
    away_rest    = away_form.get("days_rest", 4) or 4
    fatigue_flag = home_rest < 3 or away_rest < 3

    # Enrich with rolling match_stats from store (corners, cards, fouls)
    from store import get_team_rolling_stats
    home_stats = get_team_rolling_stats(home, league, window=5)
    away_stats = get_team_rolling_stats(away, league, window=5)

    # Override xG from stored match_stats if available (more accurate than form cache)
    if home_stats.get("avg_xg"): home_xg = home_stats["avg_xg"]
    if away_stats.get("avg_xg"): away_xg = away_stats["avg_xg"]

    # Rolling league goals per game — used by draw suppression in predictor
    from store import get_league_rolling_mean
    league_rolling_goals = get_league_rolling_mean(league, window=30)  # None if < 10 matches

    return {
        **fixture,
        "odds":                 odds,
        "regime":               regime,
        "drs":                  drs,
        "motivation_delta":     motiv,
        "home_xg":              home_xg,
        "away_xg":              away_xg,
        "home_xg_source":       home_xg_src,
        "away_xg_source":       away_xg_src,
        "combined_xg":          round(home_xg + away_xg, 2) if home_xg and away_xg else None,
        "home_position":        home_pos,
        "away_position":        away_pos,
        "home_rest":            home_rest,
        "away_rest":            away_rest,
        "fatigue_flag":         fatigue_flag,
        "ref_red_rate":         drs_base,
        "league_rolling_goals": league_rolling_goals,
        # From match_stats rolling window
        "home_corners_avg": home_stats.get("avg_corners"),
        "away_corners_avg": away_stats.get("avg_corners"),
        "home_foul_rate":   home_form.get("foul_rate"),
        "away_foul_rate":   away_form.get("foul_rate"),
        "derby_flag":       1 if home[:3].lower() == away[:3].lower() else 0,
        "home_n_matches":   home_stats.get("n_matches", 5),
    }


# ══════════════════════════════════════════════
# STEP 3 — Predict + risk engine
# ══════════════════════════════════════════════
def run_predictions_pipeline(
    fixtures: List[Dict],
    bankroll: float,
    min_ev: float,
    run_id: str,
) -> Tuple[List[Dict], List[Dict]]:

    brier = get_brier_scores(last_n=100)
    blend_weight = brier.get("recommended_blend", 0.60)
    n_cal = brier.get("n", 0)

    if n_cal == 0:
        logger.info("Blend weight: 0.600 (default — no calibration data yet; backfill populates automatically)")
    elif n_cal < brier.get("min_sample_for_blend", 30):
        logger.info(f"Blend weight: {blend_weight:.3f}  (default — need {brier['min_sample_for_blend'] - n_cal} more results for formula | n={n_cal})")
    else:
        stable = " [STABLE]" if brier.get("blend_stable") else ""
        logger.info(
            f"Blend weight: {blend_weight:.3f}{stable}  "
            f"(n={n_cal} | model_BS={brier['model_brier']:.4f} market_BS={brier['market_brier']:.4f})"
        )

    # Load calibration slopes — prefer model_metrics table (persistent),
    # fall back to CalibrationLab in-memory if table empty
    from store import get_latest_cal_slopes
    stored_slopes = get_latest_cal_slopes()
    cal_lab = CalibrationLab()
    cal_metrics = cal_lab.all_metrics()

    predictor = Predictor()
    predictor.blend_weight = blend_weight

    # Apply slopes: stored_slopes (from DB) take precedence over in-memory
    slopes_applied = {**{m: v.cal_slope for m, v in cal_metrics.items()},
                      **stored_slopes}  # DB overwrites in-memory
    for market, slope in slopes_applied.items():
        if abs(slope - 1.0) > 0.05:   # only apply meaningful corrections
            predictor.update_calibration(market, slope)
    # Register active signals
    from store import register_signal, get_active_signals as _gas
    for ns, sigs in [("goals",["rolling_xg","rest_days","shot_conv"]),
                     ("corners",["possession_pct"]),
                     ("cards",["ref_strictness","derby_flag","foul_rate"])]:
        for s in sigs:
            register_signal(s, ns, ns)

    risk    = RiskEngine(initial_bankroll=bankroll)
    edge_bets, all_preds = [], []

    for fx in fixtures:
        enriched = enrich_fixture(fx)
        cid      = enriched.get("canonical_id","")
        regime   = enriched["regime"]
        drs      = enriched["drs"]
        motiv    = enriched["motivation_delta"]
        odds_map = enriched.get("odds", {})

        row = {
            "home": enriched["home"], "away": enriched["away"],
            "league": enriched["league"], "regime": regime,
            "drs": drs, "motivation_delta": motiv,
            # Goals / 1X2
            "odds_1":      odds_map.get("home_win"),
            "odds_X":      odds_map.get("draw"),
            "odds_2":      odds_map.get("away_win"),
            "odds_over25": odds_map.get("over25"),
            "odds_u25":    odds_map.get("under25"),
            # Corners / cards odds (from OddsFetcher or manual feed)
            "odds_corners_over": odds_map.get("corners_over"),
            "odds_cards_over":   odds_map.get("cards_over"),
            # xG features (for Poisson lambdas + GameTempo)
            "home_xg":         enriched.get("home_xg"),
            "away_xg":         enriched.get("away_xg"),
            "home_xg_source":  enriched.get("home_xg_source", "prior"),
            "away_xg_source":  enriched.get("away_xg_source", "prior"),
            "home_n_matches":  enriched.get("home_n_matches", 5),
            # Corners features
            "home_corners_avg": enriched.get("home_corners_avg"),
            "away_corners_avg": enriched.get("away_corners_avg"),
            "corners_line":     fx.get("odds", {}).get("corners_line"),
            # Cards features
            "home_foul_rate":  enriched.get("home_foul_rate"),
            "away_foul_rate":  enriched.get("away_foul_rate"),
            "derby_flag":      enriched.get("derby_flag", 0),
            "ref_strictness":  enriched.get("ref_red_rate", 0.12) * 25,
            # Rest
            "home_rest_days":  enriched.get("home_rest"),
            # Draw context
            "league_rolling_goals": enriched.get("league_rolling_goals"),
        }
        # Remove None values
        row = {k:v for k,v in row.items() if v is not None}

        market_preds = predictor.predict(row)
        if not market_preds:
            continue

        fx_bets = []
        for pred in market_preds:
            if regime == "compression" and pred.market == "over25":
                continue
            if not pred.has_edge or pred.ev_pct < min_ev:
                continue

            signal = ErrorSignal(
                drs=drs,
                regime=regime,
                p_model_market_gap=abs(pred.p_model - pred.p_market),
                motivation_delta=motiv,
                is_compression_overs=False,
                league_sample_size=_league_sample_size(enriched["league"]),
            )
            raw_kelly = max((pred.p_true*pred.decimal_odds-1)/(pred.decimal_odds-1),0) if pred.decimal_odds>1 else 0

            # Uncertainty-adjusted Kelly: stake naturally grows as calibration n increases.
            # penalty = 1/(1+√(n/30)) → 1.0 at n=0, 0.77 at n=30, 0.59 at n=100, 0.50 at n=200+
            # Full formula: 0.35 × Kelly × uncertainty_penalty (conservative base = 35% Kelly)
            # Capped at 3% bankroll per bet regardless.
            import math as _math
            n_cal = brier.get("n", 0)
            uncertainty_penalty = 1.0 / (1.0 + _math.sqrt(n_cal / 30.0)) if n_cal > 0 else 1.0
            raw_stake = min(raw_kelly * 0.35 * uncertainty_penalty, 0.03) * bankroll
            adj = risk.adjust_stake(raw_stake, signal)

            drift = odds_map.get(f"{pred.market}_drift_pct")
            drift_note = ""
            if drift and 5 < drift < 100:   # >100% move = data artifact from stale/flipped snapshot
                drift_note = f"Odds drifted out +{drift:.1f}% (possible steam against)"
            elif drift and drift < -5:
                drift_note = f"Odds shortened {drift:.1f}% (market confirming)"

            bet = {
                "match_id":          cid,
                "home":              enriched["home"],
                "away":              enriched["away"],
                "league":            enriched["league"],
                "kickoff_utc":       enriched.get("kickoff_utc",""),
                "market":            pred.market,
                "odds":              pred.decimal_odds,
                "p_true":            pred.p_true,
                "p_true_low":        pred.p_low,
                "p_true_high":       pred.p_high,
                "p_market":          pred.p_market,
                "p_model":           pred.p_model,
                "edge":              round(pred.p_true - pred.p_market, 4),
                "ev_pct":            pred.ev_pct,
                "regime":            regime,
                "regime_delta":      pred.regime_delta,
                "drs":               drs,
                "game_tempo":        pred.game_tempo,
                "cal_slope":         pred.cal_slope,
                "motivation_delta":  motiv,
                "blend_weight":      round(blend_weight, 3),
                "stake_pct":         round(adj["final_stake"]/bankroll, 4),
                "stake_kes":         adj["final_stake"],
                "risk_reduction_pct":adj["total_reduction_pct"],
                "error_prob":        adj["error_prob"],
                "vol_regime":        adj["vol_regime"],
                "governor_active":   adj["governor_active"],
                "governor_drawdown_pct": adj["current_drawdown_pct"],
                "drift_pct":         drift,
                "drift_note":        drift_note,
                "home_xg":           enriched.get("home_xg"),
                "away_xg":           enriched.get("away_xg"),
                "combined_xg":       enriched.get("combined_xg"),
                "home_position":     enriched.get("home_position"),
                "away_position":     enriched.get("away_position"),
                "fatigue_flag":      enriched.get("fatigue_flag"),
                # CLV tracking: record placed odds so you can compare against closing line
                "clv_target_odds":   pred.decimal_odds,
                "clv_note":          "Compare placed odds against closing line before kickoff",
                # Context fields
                "motivation_delta":  motiv,
                "home_rest_days":    enriched.get("home_rest", 4),
                "away_rest_days":    enriched.get("away_rest", 4),
                "draw_adjustments":  pred.notes if pred.market == "draw" else "",
                "rationale": (
                    f"EV +{pred.ev_pct:.1f}% | "
                    f"Edge +{round((pred.p_true-pred.p_market)*100,1)}pp | "
                    f"Regime: {regime} | DRS: {drs:.2f} | "
                    f"p_true {pred.p_true:.1%} vs market {pred.p_market:.1%}"
                    + (f" | motiv_delta={motiv:+d}" if motiv != 0 else "")
                    + (f" | {pred.notes}" if pred.notes else "")
                    + (f" | {drift_note}" if drift_note else "")
                ),
            }
            fx_bets.append(bet)
            save_prediction({**bet, "p_model": pred.p_model}, run_id)

        edge_bets.extend(fx_bets)
        all_preds.append({
            "canonical_id":  cid,
            "home":          enriched["home"],
            "away":          enriched["away"],
            "league":        enriched["league"],
            "kickoff_utc":   enriched.get("kickoff_utc",""),
            "regime":        regime,
            "drs":           drs,
            "combined_xg":   enriched.get("combined_xg"),
            "home_position": enriched.get("home_position"),
            "away_position": enriched.get("away_position"),
            "fatigue_flag":  enriched.get("fatigue_flag"),
            "status":        "unplayed",
            "edge_bets":     fx_bets,
            "has_any_edge":  len(fx_bets) > 0,
        })

    edge_bets.sort(key=lambda x: x["ev_pct"], reverse=True)
    logger.info(f"Pipeline: {len(all_preds)} fixtures → {len(edge_bets)} edge bets")
    return edge_bets, all_preds


def _league_sample_size(league: str) -> int:
    try:
        from store import _conn
        with _conn() as con:
            r = con.execute("""
                SELECT COUNT(*) as n FROM results r
                JOIN fixtures f ON r.canonical_id=f.canonical_id
                WHERE f.league=?
            """, (league,)).fetchone()
        return r["n"] if r else 0
    except Exception:
        return 0


# ══════════════════════════════════════════════
# STEP 4 — Portfolio → JSON
# ══════════════════════════════════════════════
def build_portfolio_json(edge_bets: List[Dict], bankroll: float) -> Optional[Dict]:
    if not edge_bets:
        return None

    import pandas as pd
    df = pd.DataFrame(edge_bets)
    if "decimal_odds" not in df.columns:
        df["decimal_odds"] = df["odds"]

    builder   = PortfolioBuilder(bankroll=bankroll)
    portfolio = builder.build(df)

    if not portfolio.singles:
        return None

    def _s(s):
        # Look up full bet detail from edge_bets (has p_market, risk fields, etc.)
        src = next((b for b in edge_bets
                    if b["home"]==s.home and b["away"]==s.away and b["market"]==s.market), {})
        p_market = src.get("p_market", round(1/s.decimal_odds*0.952, 4))
        return {
            "home":          s.home,
            "away":          s.away,
            "league":        s.league,
            "kickoff_utc":   src.get("kickoff_utc",""),
            "market":        s.market,
            "odds":          s.decimal_odds,
            "p_true":        s.p_true,
            "p_true_band":   [src.get("p_true_low", round(s.p_true-0.10,3)),
                              src.get("p_true_high", round(s.p_true+0.10,3))],
            "p_market":      p_market,
            "p_model":       src.get("p_model", s.p_true),
            "edge":          round(s.p_true - p_market, 4),
            "ev_pct":        s.ev_pct,
            "regime":        s.regime,
            "drs":           s.drs,
            "stake_pct":     s.stake_units,
            "stake_kes":     s.stake_kes,
            "return_if_win": round(s.stake_kes*s.decimal_odds, 2),
            "profit_if_win": round(s.stake_kes*(s.decimal_odds-1), 2),
            "rationale":     s.rationale,
            "drift_note":    src.get("drift_note",""),
            "combined_xg":   src.get("combined_xg"),
            "home_position": src.get("home_position"),
            "away_position": src.get("away_position"),
            "fatigue_flag":  src.get("fatigue_flag"),
            "risk_flags": {
                "error_prob":         src.get("error_prob", 0),
                "risk_reduction_pct": src.get("risk_reduction_pct", 0),
                "vol_regime":         src.get("vol_regime",""),
                "governor_active":    src.get("governor_active", False),
                "governor_drawdown":  src.get("governor_drawdown_pct", 0),
            },
        }

    singles_out = [_s(s) for s in portfolio.singles]

    parlay_out = None
    if portfolio.parlay and portfolio.parlay.is_valid:
        p = portfolio.parlay
        parlay_out = {
            "legs":             len(p.legs),
            "combined_odds":    p.combined_odds,
            "combined_p_true":  p.combined_p_true,
            "combined_ev_pct":  p.combined_ev_pct,
            "stake_pct":        p.stake_units,
            "stake_kes":        p.stake_kes,
            "return_if_win":    round(p.stake_kes*p.combined_odds, 2),
            "profit_if_win":    round(p.stake_kes*(p.combined_odds-1), 2),
            "compression_flag": p.compression_flag,
            "size_reduction":   p.size_reduction,
            "drs_flag":         p.drs_flag,
            "veto_reason":      "",
            "legs_detail": [
                {
                    "home":        l.home,
                    "away":        l.away,
                    "league":      l.league,
                    "kickoff_utc": next((b["kickoff_utc"] for b in edge_bets
                                         if b["home"]==l.home and b["away"]==l.away and b["market"]==l.market),""),
                    "market":      l.market,
                    "odds":        l.decimal_odds,
                    "p_true":      l.p_true,
                    "p_market":    next((b["p_market"] for b in edge_bets
                                         if b["home"]==l.home and b["market"]==l.market), None),
                    "ev_pct":      l.ev_pct,
                    "regime":      l.regime,
                    "drs":         l.drs,
                }
                for l in p.legs
            ],
        }

    st = sum(s["stake_kes"] for s in singles_out)
    pt = parlay_out["stake_kes"] if parlay_out else 0
    te = st + pt

    return {
        "singles": singles_out,
        "parlay":  parlay_out,
        "totals": {
            "singles_stake_kes":  st,
            "parlay_stake_kes":   pt,
            "total_exposure_kes": te,
            "total_exposure_pct": round(te/bankroll*100, 2),
            "max_return_kes":     round(sum(s["return_if_win"] for s in singles_out)+(parlay_out["return_if_win"] if parlay_out else 0), 2),
            "worst_case_kes":     -round(te, 2),
            "theoretical_ev_kes": round(
                sum(s["ev_pct"]/100*s["stake_kes"] for s in singles_out)+
                ((parlay_out["combined_ev_pct"]/100*parlay_out["stake_kes"]) if parlay_out else 0), 2),
        },
        "risk_flags": {
            "drawdown_governor":          "inactive",
            "compression_overs_filtered": True,
            "team_overlap_clean":         True,
            "blend_weight_source":        "stored_brier_scores",
        },
    }


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def run(
    bankroll: float = BANKROLL_DEFAULT,
    min_ev:   float = EV_THRESHOLD,
    target_date: Optional[str] = None,
    lookahead:   int = 2,
) -> Dict:
    init_db()

    now    = datetime.now(timezone.utc)
    run_id = f"V41_{now.strftime('%Y%m%d_%H%M%S')}"
    today  = target_date or now.strftime("%Y-%m-%d")

    logger.info(f"\n{'='*60}")
    logger.info(f"  Vantage Engine v4.1 | {run_id}")
    logger.info(f"  {today} | Bankroll: KES {bankroll:,.0f}")
    logger.info(storage_summary())
    logger.info(f"{'='*60}")

    # ── Backfill historical results for regime/rolling mean training ──────────
    # Runs on every session start. Fetches completed ESPN matches for last 7 days
    # and writes scores to league_stats. Safe to run repeatedly (INSERT OR IGNORE).
    # Takes ~5-10s. Silently skips if no network.
    try:
        n_backfilled = backfill_results_from_espn(days_back=7)
        if n_backfilled > 0:
            logger.info(f"Backfill: {n_backfilled} historical results loaded into league_stats")
    except Exception as e:
        logger.debug(f"Backfill skipped (non-fatal): {e}")

    # ── Brier calibration backfill ────────────────────────────────────────────
    # Join predictions + league_stats scores → populate results table with outcomes.
    # Runs every session (INSERT OR IGNORE = safe). Unlocks Brier blend weight
    # as soon as predictions + scores are both present.
    try:
        brier_backfill = backfill_brier_from_league_stats()
        if brier_backfill["inserted"] > 0:
            logger.info(f"Brier backfill: {brier_backfill['inserted']} new calibration results added")
    except Exception as e:
        logger.debug(f"Backfill skipped (non-fatal): {e}")

    fixtures, from_cache = fetch_or_load_fixtures(target_date=target_date, lookahead=lookahead)
    if not fixtures:
        output = {"meta": {"run_id":run_id,"date":today,"bankroll_kes":bankroll,
                           "fixtures_scanned":0,"total_edge_bets":0,
                           "reason":"No unplayed fixtures"},
                  "output_slip":None,"all_edge_bets":[],"all_predictions":[]}
        _save(output, run_id)
        return output

    brier        = get_brier_scores()

    # ── Auto-enrich team form: FBref (real xG) → ESPN fallback ───────────────
    # Stage 1: FBref gives real xG for top leagues (slow, ~5s/team, Playwright)
    # Stage 2: ESPN gives shots-proxy xG for all leagues (fast, no browser)
    # Both skip if cache fresh (<12h). Failures are non-fatal.
    with _conn() as con:
        total_fx = con.execute(
            "SELECT COUNT(DISTINCT home) + COUNT(DISTINCT away) as n "
            "FROM fixtures WHERE match_date=? AND status='unplayed'", (today,)
        ).fetchone()["n"] or 1
        # Count only today's fixture teams that have fresh, sourced xG
        # (xg_source NULL = pre-migration entry, don't count as warm)
        cached_xg = con.execute("""
            SELECT COUNT(*) as n FROM (
                SELECT DISTINCT tf.team FROM team_form tf
                JOIN fixtures fx ON (tf.team = fx.home OR tf.team = fx.away)
                WHERE fx.match_date=? AND fx.status='unplayed'
                  AND tf.expires_at > ? AND tf.avg_xg_for IS NOT NULL
                  AND tf.xg_source IS NOT NULL
            )
        """, (today, now.isoformat())).fetchone()["n"] or 0
    cache_pct = cached_xg / max(total_fx, 1)

    if cache_pct >= 0.75:
        logger.info(f"Form cache warm ({cached_xg}/{total_fx} teams with sourced xG) — skipping enrichment")
    else:
        logger.info(f"Form cache cold ({cached_xg}/{total_fx} teams) — running enrichers")

        # Stage 1: Understat real xG (EPL/La Liga/Bundesliga/Serie A/Ligue 1)
        # One page load per league covers all ~20 teams. No CF, no API key.
        try:
            from understat_enricher import enrich_todays_fixtures as understat_enrich
            understat_enrich(match_date=today)
            logger.info("Understat enrichment complete")
        except Exception as e:
            logger.warning(f"Understat enricher failed (non-fatal): {e}")

        # Stage 2: Sofascore — Americas, Asia, Middle East (leagues ESPN has no stats for)
        try:
            from sofascore_enricher import enrich_todays_fixtures as sofascore_enrich
            sofascore_enrich(match_date=today)
            logger.info("Sofascore enrichment complete")
        except Exception as e:
            logger.warning(f"Sofascore enricher failed (non-fatal): {e}")

        # Stage 3: ESPN fallback — remaining leagues (shots-proxy xG)
        try:
            from espn_enricher import enrich_todays_fixtures as espn_enrich
            espn_enrich(match_date=today)
            logger.info("ESPN enrichment complete")
        except Exception as e:
            logger.warning(f"ESPN enricher failed (non-fatal): {e}")

    edge_bets, all_preds = run_predictions_pipeline(fixtures, bankroll, min_ev, run_id)
    portfolio    = build_portfolio_json(edge_bets, bankroll)

    if portfolio:
        save_portfolio_predictions(portfolio, run_id)

    brier        = get_brier_scores(last_n=100)
    _blend       = brier.get("recommended_blend", 0.60)

    output = {
        "meta": {
            "run_id":              run_id,
            "generated_utc":       now.isoformat(),
            "date":                today,
            "bankroll_kes":        bankroll,
            "ev_threshold_pct":    min_ev,
            "fixtures_scanned":    len(fixtures),
            "fixtures_from_cache": from_cache,
            "fixtures_with_edge":  sum(1 for p in all_preds if p["has_any_edge"]),
            "total_edge_bets":     len(edge_bets),
            "blend_weight":        edge_bets[0]["blend_weight"] if edge_bets else round(_blend, 3),
            "brier_calibration":   brier,
        },
        "output_slip":     portfolio,
        "all_edge_bets":   edge_bets,
        "all_predictions": all_preds,
    }

    _save(output, run_id)
    _print_summary(output)
    return output


def _save(output, run_id):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    for path in [f"predictions_{ts}.json", "predictions_latest.json"]:
        with open(path,"w") as f:
            json.dump(output, f, indent=2, default=str)
    logger.info("Saved: predictions_latest.json")


def _print_summary(output):
    meta = output["meta"]
    slip = output.get("output_slip")
    print(f"\n{'═'*60}")
    print(f"  VANTAGE ENGINE — {meta['date']}  |  {meta['run_id']}")
    print(f"{'═'*60}")
    print(f"  Fixtures: {meta['fixtures_scanned']} ({'cache' if meta.get('fixtures_from_cache') else 'live'})")
    print(f"  Edge bets: {meta['total_edge_bets']}  |  Blend: {meta['blend_weight']:.3f}")
    if not slip:
        print("  No output slip — insufficient edge bets")
        print(f"{'═'*60}\n")
        return
    print(f"\n  SINGLES")
    for i, s in enumerate(slip["singles"], 1):
        print(f"  [{i}] {s['home']} vs {s['away']}  ({s['league']})")
        mkt = s['market'].replace('_',' ').title()
        print(f"       {mkt} @ {s['odds']} | p_true {s['p_true']:.1%} | "
              f"EV +{s['ev_pct']:.1f}% | Stake KES {s['stake_kes']:,.0f}")
        # Draw-specific warnings
        if s.get("market") == "draw":
            print(f"       [DRAW] High-variance market — needs real edge to be worth it")
            if s.get("draw_adjustments"):
                print(f"       [DRAW] Adjustments applied: {s['draw_adjustments']}")
        # Motivation context
        motiv = s.get("motivation_delta", 0)
        if motiv != 0:
            direction = "away team more motivated" if motiv > 0 else "home team more motivated"
            print(f"       Motivation delta: {motiv:+d} ({direction})")
        # Fatigue context
        hr = s.get("home_rest_days", 4)
        ar = s.get("away_rest_days", 4)
        if hr < 3 or ar < 3:
            print(f"       Fatigue: home {hr:.0f}d rest / away {ar:.0f}d rest")
        # CLV target
        print(f"       CLV target: {s['odds']} — check closing line before kickoff")
        # Corners line mismatch warning
        if s.get("market") == "corners_over":
            cline = s.get("corners_line_used", 9.5)
            print(f"       ⚠ Corners line: {cline} — verify this matches Betika's actual line")
        if s.get("drift_note"):
            print(f"       {s['drift_note']}")
    if slip["parlay"]:
        p = slip["parlay"]
        print(f"\n  PARLAY ({p['legs']} legs | {p['combined_odds']:.2f}x | EV +{p['combined_ev_pct']:.1f}%)")
        for i, l in enumerate(p["legs_detail"], 1):
            print(f"  [{i}] {l['home']} vs {l['away']} — {l['market'].replace('_',' ').title()} @ {l['odds']}")
        print(f"       Stake KES {p['stake_kes']:,.0f}  ->  Returns KES {p['return_if_win']:,.0f}")
    t = slip["totals"]
    print(f"\n  TOTAL: KES {t['total_exposure_kes']:,.0f} ({t['total_exposure_pct']:.1f}%)")
    print(f"  -> predictions_latest.json")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
def cmd_settle_all(json_path: str = "predictions_latest.json", dry_run: bool = False):
    """
    Auto-settle all predictions in a predictions JSON file.

    For each prediction:
      1. Look up the score from league_stats (matched by home/away/date).
      2. Derive outcome (won=1 / lost=0) for the market.
      3. Call settle_result() and update CalibrationLab.

    Markets auto-derived from scoreline:
      home_win, draw, away_win, over25, under25, over15, btts_yes, btts_no

    Markets that need manual settle (no scoreline data):
      corners_over, cards_over — printed as a reminder.

    Usage:
      python run_predictions.py settle-all
      python run_predictions.py settle-all predictions_20260311_1511.json
      python run_predictions.py settle-all --dry-run   (preview without writing)
    """
    import json as _json
    from calibration import CalibrationLab
    from store import save_model_metrics, _conn as _store_conn

    init_db()

    import glob as _glob, os as _os

    # If default file has no edge bets, search for most recent file that does
    def _load_with_bets(path):
        try:
            with open(path) as f:
                d = _json.load(f)
            if d.get("all_edge_bets"):
                return d
        except Exception:
            pass
        return None

    data = _load_with_bets(json_path)
    if not data:
        # Try all timestamped prediction files, most recent first
        candidates = sorted(
            _glob.glob("predictions_*.json"),
            key=_os.path.getmtime, reverse=True
        )
        for c in candidates:
            data = _load_with_bets(c)
            if data:
                print(f"  Using {c} (predictions_latest.json had no edge bets)")
                break

    if not data:
        print("No predictions file with edge bets found.")
        print("Files checked: predictions_latest.json + predictions_*.json")
        return

    # all_edge_bets has one row per market with match_id, market, odds, p_true, stake_kes
    all_preds = data.get("all_edge_bets", [])
    if not all_preds:
        print("No edge bets found in any predictions file.")
        return

    settled = 0
    skipped_no_score = 0
    skipped_market = 0
    skipped_already = 0
    voided = []
    manual_needed = []

    cal = CalibrationLab()

    with _store_conn() as con:
        # Pre-load already-settled canonical_id+market pairs to avoid duplicates
        already_settled = {
            (r["canonical_id"], r["market"])
            for r in con.execute("SELECT canonical_id, market FROM results").fetchall()
        }

    for pred in all_preds:
        cid    = pred.get("match_id") or pred.get("canonical_id", "")
        market = pred.get("market", "")
        odds   = float(pred.get("odds") or pred.get("decimal_odds") or 0)
        p_true = pred.get("p_true", 0)
        stake  = float(pred.get("stake_kes") or 0)
        regime = pred.get("regime", "neutral")

        if not cid or not market:
            continue

        # Corners/cards — look up from match_stats (populated by backfill)
        if market in ("corners_over", "corners_under", "cards_over", "cards_under"):
            with _store_conn() as con:
                ms = con.execute("""
                    SELECT corners_home, corners_away, cards_home, cards_away
                    FROM match_stats WHERE canonical_id = ?
                    LIMIT 1
                """, (cid,)).fetchone()
                # Fallback: join via fixtures → match_stats by home/away/date
                if not ms and home_team and away_team:
                    ms = con.execute("""
                        SELECT ms.corners_home, ms.corners_away,
                               ms.cards_home, ms.cards_away
                        FROM match_stats ms
                        JOIN fixtures f ON ms.canonical_id = f.canonical_id
                        WHERE LOWER(TRIM(f.home)) = LOWER(TRIM(?))
                          AND LOWER(TRIM(f.away)) = LOWER(TRIM(?))
                          AND f.match_date = ?
                        LIMIT 1
                    """, (home_team, away_team, match_date_fmt)).fetchone()

            if not ms or (ms["corners_home"] is None and ms["cards_home"] is None):
                manual_needed.append(f"  {cid}  {market}  @ {odds}  (no stats in DB yet)")
                skipped_no_score += 1
                continue

            if market in ("corners_over", "corners_under"):
                if ms["corners_home"] is None:
                    manual_needed.append(f"  {cid}  {market}  @ {odds}  (corners not in DB)")
                    skipped_no_score += 1
                    continue
                total_corners = ms["corners_home"] + ms["corners_away"]
                # Get the line from the prediction's placed odds context
                # Use the corners_line stored in the edge bet if available, else 9.5
                corners_line = float(pred.get("corners_line_used") or pred.get("corners_line") or 9.5)
                if market == "corners_over":
                    outcome = 1 if total_corners > corners_line else 0
                else:
                    outcome = 1 if total_corners <= corners_line else 0
                result_detail = f"{ms['corners_home']}+{ms['corners_away']}={total_corners} vs line {corners_line}"

            elif market in ("cards_over", "cards_under"):
                if ms["cards_home"] is None:
                    manual_needed.append(f"  {cid}  {market}  @ {odds}  (cards not in DB)")
                    skipped_no_score += 1
                    continue
                total_cards = ms["cards_home"] + ms["cards_away"]
                cards_line = float(pred.get("cards_line_used") or 3.5)
                if market == "cards_over":
                    outcome = 1 if total_cards > cards_line else 0
                else:
                    outcome = 1 if total_cards <= cards_line else 0
                result_detail = f"{ms['cards_home']}+{ms['cards_away']}={total_cards} vs line {cards_line}"

            pnl = round((odds - 1) * stake if outcome else -stake, 2)
            result_str = "WON " if outcome else "LOSS"
            if dry_run:
                print(f"  [DRY] {cid:<30} {market:<14} {result_str}  {result_detail}  pnl={pnl:+.0f}")
                settled += 1
                continue
            settle_result(
                canonical_id=cid, market=market, outcome=outcome,
                closing_odds=odds, placed_odds=odds,
                stake_kes=stake, pnl_kes=pnl,
            )
            cal.record(market, p_true, outcome, regime=regime)
            metrics = cal.compute_metrics(market)
            save_model_metrics({market: metrics.__dict__})
            already_settled.add((cid, market))
            settled += 1
            print(f"  ✓ {cid:<30} {market:<14} {result_str}  {result_detail}  pnl={pnl:+.0f}")
            continue

        # Skip already settled
        if (cid, market) in already_settled:
            skipped_already += 1
            continue

        # Look up score from league_stats
        # Try direct league_stats match via canonical_id → fixture home/away/date first,
        # then fall back to matching by home/away name extracted from canonical_id date suffix
        match_date = cid[-8:]  # last 8 chars = YYYYMMDD
        match_date_fmt = f"{match_date[:4]}-{match_date[4:6]}-{match_date[6:8]}"
        home_team = pred.get("home", "")
        away_team = pred.get("away", "")
        with _store_conn() as con:
            # Primary: join via fixtures table
            row = con.execute("""
                SELECT ls.home_goals, ls.away_goals, ls.total_goals
                FROM fixtures f
                JOIN league_stats ls
                  ON LOWER(TRIM(ls.home_team)) = LOWER(TRIM(f.home))
                 AND LOWER(TRIM(ls.away_team)) = LOWER(TRIM(f.away))
                 AND ls.match_date = f.match_date
                WHERE f.canonical_id = ?
                  AND ls.home_goals IS NOT NULL
                LIMIT 1
            """, (cid,)).fetchone()
            # Fallback: direct league_stats match by home/away/date
            if not row and home_team and away_team:
                row = con.execute("""
                    SELECT home_goals, away_goals, total_goals
                    FROM league_stats
                    WHERE LOWER(TRIM(home_team)) = LOWER(TRIM(?))
                      AND LOWER(TRIM(away_team)) = LOWER(TRIM(?))
                      AND match_date = ?
                      AND home_goals IS NOT NULL
                    LIMIT 1
                """, (home_team, away_team, match_date_fmt)).fetchone()

        # If not in league_stats yet (same-day match), try ESPN live scoreboard
        if not row and home_team and away_team:
            try:
                import urllib.request, json as _espn_json
                from normalizer import normalize_team as _nt
                ESPN_LEAGUES_SETTLE = {
                    "Premier League": "eng.1", "Championship": "eng.2",
                    "League One": "eng.3", "League Two": "eng.4",
                    "La Liga": "esp.1", "Segunda Division": "esp.2",
                    "Bundesliga": "ger.1", "2. Bundesliga": "ger.2",
                    "Serie A": "ita.1", "Serie B": "ita.2",
                    "Ligue 1": "fra.1", "Ligue 2": "fra.2",
                    "Eredivisie": "ned.1", "Primeira Liga": "por.1",
                    "Süper Lig": "tur.1", "Scottish Premiership": "sco.1",
                    "Belgian Pro League": "bel.1", "Danish Superliga": "den.1",
                    "Russian Premier League": "rus.1",
                    "Champions League": "uefa.champions",
                    "Europa League": "uefa.europa",
                    "Conference League": "uefa.europa.conf",
                    "Liga MX": "mex.1", "MLS": "usa.1",
                    "Brasileirao": "bra.1", "Argentine Primera": "arg.1",
                    "Saudi Pro League": "ksa.1", "A-League": "aus.1",
                    # International
                    "UEFA World Cup Qualifiers":     "fifa.worldq.uefa",
                    "CONMEBOL World Cup Qualifiers": "fifa.worldq.conmebol",
                    "CONCACAF World Cup Qualifiers": "fifa.worldq.concacaf",
                    "CAF World Cup Qualifiers":      "fifa.worldq.caf",
                    "AFC World Cup Qualifiers":      "fifa.worldq.afc",
                    "UEFA Nations League":           "uefa.nations",
                    "CONCACAF Nations League":       "concacaf.nations",
                    "Copa America":                  "conmebol.america",
                    "AFCON":                         "caf.nations",
                    "AFC Asian Cup":                 "afc.cup",
                    "International Friendlies":      "fifa.friendly",
                }
                league_name = pred.get("league", "")
                code = ESPN_LEAGUES_SETTLE.get(league_name)
                if code:
                    url = (f"https://site.api.espn.com/apis/site/v2/sports/soccer/"
                           f"{code}/scoreboard?dates={match_date}")
                    req = urllib.request.Request(url, headers={"User-Agent": "VantageSettle/1.0"})
                    with urllib.request.urlopen(req, timeout=8) as resp:
                        espn_data = _espn_json.loads(resp.read())
                    for event in espn_data.get("events", []):
                        comp = event["competitions"][0]
                        if comp.get("status", {}).get("type", {}).get("state") != "post":
                            continue
                        competitors = comp.get("competitors", [])
                        if len(competitors) < 2:
                            continue
                        h_obj = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                        a_obj = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
                        h_name = _nt(h_obj["team"]["displayName"])
                        a_name = _nt(a_obj["team"]["displayName"])
                        from fuzzywuzzy import fuzz
                        if (fuzz.ratio(h_name.lower(), home_team.lower()) > 70 and
                                fuzz.ratio(a_name.lower(), away_team.lower()) > 70):
                            hg_live = int(h_obj.get("score", 0) or 0)
                            ag_live = int(a_obj.get("score", 0) or 0)
                            row = {"home_goals": hg_live, "away_goals": ag_live,
                                   "total_goals": hg_live + ag_live}
                            break
            except Exception:
                pass

        if not row:
            # Auto-void if match was >48h ago and still no score — stale pending
            match_dt_str = match_date_fmt  # YYYY-MM-DD
            try:
                from datetime import date as _date
                match_day = _date(int(match_dt_str[:4]), int(match_dt_str[5:7]), int(match_dt_str[8:10]))
                days_old = (_date.today() - match_day).days
                if days_old > 2:
                    voided.append(f"  {cid:<30} {market:<14} VOID  ({days_old}d old, no score found)")
                    skipped_no_score += 1
                else:
                    skipped_no_score += 1
            except Exception:
                skipped_no_score += 1
            continue

        hg = row["home_goals"]
        ag = row["away_goals"]
        tg = row["total_goals"] or (hg + ag)

        # Derive outcome
        outcome = None
        if market == "home_win":   outcome = 1 if hg > ag else 0
        elif market == "draw":     outcome = 1 if hg == ag else 0
        elif market == "away_win": outcome = 1 if ag > hg else 0
        elif market == "over25":   outcome = 1 if tg > 2 else 0
        elif market == "under25":  outcome = 1 if tg <= 2 else 0
        elif market == "over15":   outcome = 1 if tg > 1 else 0
        elif market == "btts_yes": outcome = 1 if (hg > 0 and ag > 0) else 0
        elif market == "btts_no":  outcome = 1 if not (hg > 0 and ag > 0) else 0

        if outcome is None:
            skipped_market += 1
            continue

        pnl = round((odds - 1) * stake if outcome else -stake, 2)
        result_str = "WON " if outcome else "LOSS"

        if dry_run:
            print(f"  [DRY] {cid:<30} {market:<12} {result_str}  {hg}-{ag}  pnl={pnl:+.0f}")
            settled += 1
            continue

        settle_result(
            canonical_id=cid, market=market, outcome=outcome,
            closing_odds=odds, placed_odds=odds,
            stake_kes=stake, pnl_kes=pnl,
        )

        # Update CalibrationLab
        cal.record(market, p_true, outcome, regime=regime)
        metrics = cal.compute_metrics(market)
        save_model_metrics({market: metrics.__dict__})

        already_settled.add((cid, market))
        settled += 1
        print(f"  ✓ {cid:<30} {market:<12} {result_str}  {hg}-{ag}  pnl={pnl:+.0f}")

    print(f"\n  Settled: {settled}")
    print(f"  Already settled: {skipped_already}")
    print(f"  No score yet: {skipped_no_score}")
    if voided:
        print(f"\n  Auto-voided (>48h, no score found):")
        for v in voided:
            print(v)
        print(f"  → These bets cannot be settled automatically. Check results manually.")
    if manual_needed:
        print(f"\n  Corners/cards — settle manually:")
        for m in manual_needed:
            print(m)
        print(f"\n  python run_predictions.py settle CANONICAL_ID corners_over 1 CLOSING_ODDS PLACED_ODDS STAKE PNL")

    if settled > 0 and not dry_run:
        brier = get_brier_scores()
        print(f"\n  Brier calibration: n={brier['n']}  blend={brier['recommended_blend']:.3f}")


def cmd_settle(args):
    init_db()
    outcome = int(args.outcome)
    settle_result(
        canonical_id=args.canonical_id, market=args.market,
        outcome=outcome, closing_odds=float(args.closing_odds),
        placed_odds=float(args.placed_odds), stake_kes=float(args.stake),
        pnl_kes=float(args.pnl),
    )
    # Record in CalibrationLab for Brier/ECE/IR tracking
    from store import save_model_metrics
    with _conn() as con:
        pr = con.execute(
            "SELECT p_true, regime FROM predictions WHERE canonical_id=? AND market=? "
            "ORDER BY generated_at DESC LIMIT 1",
            (args.canonical_id, args.market)
        ).fetchone()
    if pr:
        cal = CalibrationLab()
        cal.record(args.market, pr["p_true"], outcome, regime=pr["regime"] or "neutral")
        metrics = cal.compute_metrics(args.market)
        save_model_metrics({args.market: metrics.__dict__})
        logger.info(f"Calibration updated [{args.market}]: "
                    f"Brier={metrics.brier_score:.4f} IR={metrics.rolling_ir:.4f} slope={metrics.cal_slope:.3f}")
    win_str = "WIN" if outcome else "LOSS"
    print(f"Result saved: {args.canonical_id} {args.market} → {win_str}")

def cmd_stats():
    init_db()
    print(storage_summary())

    clv = get_clv_summary()
    if clv.get("n"):
        print(f"\n  CLV (last {clv['n']} bets)")
        print(f"  Beat close:  {clv['clv_rate']*100:.1f}%  (target 52%)")
        print(f"  Mean CLV:    {clv['mean_clv']:+.4f}")
        print(f"  Hit rate:    {clv['hit_rate']*100:.1f}%")
        print(f"  Total PnL:   KES {clv['total_pnl']:,.0f}")

    brier = get_brier_scores()
    print(f"\n  BRIER CALIBRATION (n={brier['n']})")
    if brier["n"] == 0:
        print(f"  No calibration data yet.")
        print(f"  → Backfill runs automatically each session (needs predictions + scores to match).")
        print(f"  → Need {brier['min_sample_for_blend']}+ results for blend formula, 200+ for stability.")
    else:
        if brier["model_brier"] is not None:
            print(f"  Model BS:    {brier['model_brier']:.5f}")
            print(f"  Market BS:   {brier['market_brier']:.5f}")
        stable = " [STABLE]" if brier["blend_stable"] else f" (need {max(0, 200 - brier['n'])} more for stable)"
        print(f"  Blend:       {brier['recommended_blend']:.3f}{stable}")
        print(f"  Note:        {brier['note']}")

    markets = get_hit_rate_by_market()
    if markets:
        print(f"\n  BY MARKET (hit rate & ROI)")
        for m in markets:
            print(f"  {m['market']:<12} n={m['n']:>4}  hit={m['hit_rate']*100:.1f}%  roi={((m['roi'] or 0)*100):.1f}%")

    brier_by_mkt = get_brier_by_market()
    if brier_by_mkt:
        print(f"\n  BRIER BY MARKET (model edge vs market)")
        for m in brier_by_mkt:
            edge_str = f"+{m['model_edge']:.4f}" if m['beating'] else f"{m['model_edge']:.4f}"
            beat_icon = "✓" if m['beating'] else "✗"
            print(f"  {beat_icon} {m['market']:<12} n={m['n']:>3}  model={m['model_brier']:.4f}  market={m['market_brier']:.4f}  edge={edge_str}")

    # ── Calibration Lab — full report ─────────────────────────────────────────
    from calibration import CalibrationLab
    cal = CalibrationLab()
    print(cal.report())

    # ── Stored calibration slopes from model_metrics table ────────────────────
    from store import get_latest_cal_slopes
    slopes = get_latest_cal_slopes()
    if slopes:
        print("  CALIBRATION SLOPES (from model_metrics)")
        for mkt, slope in sorted(slopes.items()):
            flag = "  ← overconfident" if slope < 0.85 else "  ← underconfident" if slope > 1.15 else ""
            print(f"  {mkt:<12} slope={slope:.4f}{flag}")

    # ── Active signals per namespace ──────────────────────────────────────────
    from store import get_active_signals
    print("\n  ACTIVE SIGNALS")
    for ns in ["goals", "corners", "cards"]:
        sigs = get_active_signals(ns)
        print(f"  [{ns}]  {', '.join(sigs) if sigs else '(none registered yet)'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vantage Engine v4.1")
    parser.add_argument("--bankroll",  type=float, default=BANKROLL_DEFAULT)
    parser.add_argument("--min-ev",    type=float, default=EV_THRESHOLD)
    parser.add_argument("--date",      type=str,   default=None,
                        help="Target date YYYY-MM-DD (default: today UTC)")
    parser.add_argument("--lookahead", type=int,   default=2,
                        help="Days ahead to search if today has no games (default: 2)")
    sub = parser.add_subparsers(dest="cmd")

    p_settle = sub.add_parser("settle")
    p_settle.add_argument("canonical_id")
    p_settle.add_argument("market")
    p_settle.add_argument("outcome")
    p_settle.add_argument("closing_odds")
    p_settle.add_argument("placed_odds")
    p_settle.add_argument("stake")
    p_settle.add_argument("pnl")

    p_settle_all = sub.add_parser("settle-all")
    p_settle_all.add_argument("file", nargs="?", default="predictions_latest.json",
                              help="Path to predictions JSON (default: predictions_latest.json)")
    p_settle_all.add_argument("--dry-run", action="store_true",
                              help="Preview outcomes without writing to DB")

    sub.add_parser("stats")

    args = parser.parse_args()
    if args.cmd == "settle":
        cmd_settle(args)
    elif args.cmd == "settle-all":
        cmd_settle_all(json_path=args.file, dry_run=args.dry_run)
    elif args.cmd == "stats":
        cmd_stats()
    else:
        run(bankroll=args.bankroll, min_ev=args.min_ev,
            target_date=args.date, lookahead=args.lookahead)

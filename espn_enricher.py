"""
espn_enricher.py — Vantage v4.1
================================
Fetches real team data from ESPN free APIs.

Confirmed working ESPN endpoints (as of 2026-03-07):
  ✓ /teams          → team IDs (20/18 per league)
  ✓ /teams/{id}/statistics  → goals, shots (tested)
  ✓ /teams/{id}/schedule    → completed matches (tested)
  ✗ /standings      → returns {} (geo-blocked for KE)

Strategy: use /teams to get IDs, then per-team stats + schedule.
League position is set to None until a working standings source is found.

Usage:
    python espn_enricher.py                    # enrich today's leagues
    python espn_enricher.py eng.1 esp.1        # specific leagues
    python espn_enricher.py --debug eng.1      # verbose: print all ESPN keys/values
    python espn_enricher.py --refresh          # clear cache first
"""

import json
import logging
import random
import time
import urllib.request
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from store import init_db, save_team_form, get_cached_team_form, _conn

logger = logging.getLogger("Vantage.Enricher")

ESPN_BASE       = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_TEAMS      = ESPN_BASE + "/{league}/teams"
ESPN_TEAM_STATS = ESPN_BASE + "/{league}/teams/{team_id}/statistics"
ESPN_SCHEDULE   = ESPN_BASE + "/{league}/teams/{team_id}/schedule"

ENRICH_LEAGUES = {
    "eng.1": "Premier League",      "eng.2": "Championship",
    "eng.3": "League One",          "eng.4": "League Two",
    "esp.1": "La Liga",             "esp.2": "Segunda Division",
    "ger.1": "Bundesliga",          "ger.2": "2. Bundesliga",
    "ita.1": "Serie A",             "ita.2": "Serie B",
    "fra.1": "Ligue 1",             "fra.2": "Ligue 2",
    "ned.1": "Eredivisie",          "por.1": "Primeira Liga",
    "tur.1": "Süper Lig",           "sco.1": "Scottish Premiership",
    "sco.2": "Scottish Championship","bel.1": "Belgian Pro League",
    "gre.1": "Super League Greece", "den.1": "Danish Superliga",
    "sui.1": "Swiss Super League",  "aut.1": "Austrian Bundesliga",
    "cze.1": "Czech First League",  "rus.1": "Russian Premier League",
    "usa.1": "MLS",                 "mex.1": "Liga MX",
    "bra.1": "Brasileirao",         "arg.1": "Argentine Primera",
    "ksa.1": "Saudi Pro League",    "aus.1": "A-League",
    "jpn.1": "J1 League",           "chn.1": "Chinese Super League",
    "uefa.champions": "Champions League",
    "uefa.europa":    "Europa League",
    "uefa.europa.conf": "Conference League",
    "eng.fa":              "FA Cup",
    "eng.league_cup":      "EFL Cup",
    "esp.copa_del_rey":    "Copa del Rey",
    "ger.dfb_pokal":       "DFB-Pokal",
    "ita.coppa_italia":    "Coppa Italia",
    "fra.coupe_de_france": "Coupe de France",
    "ned.knvb":            "KNVB Cup",
    "por.taca":            "Taça de Portugal",
    "sco.fa":              "Scottish FA Cup",
    "sco.league_cup":      "Scottish League Cup",
    "tur.cup":             "Turkish Cup",
    "bel.cup":             "Belgian Cup",
    "gre.cup":             "Greek Cup",
    "bra.copa":            "Copa do Brasil",
    "arg.copa":            "Copa Argentina",
    "usa.open":            "US Open Cup",
}

_DEBUG = False


# ── HTTP ──────────────────────────────────────────────────────────────────────
def _get(url: str, retries: int = 2) -> Optional[Dict]:
    for attempt in range(retries + 1):
        try:
            time.sleep(random.uniform(0.5, 1.0))
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if attempt == retries:
                logger.warning(f"ESPN [{url[:65]}]: {e}")
            else:
                time.sleep(1.5 ** attempt)
    return None


def _norm(name: str) -> str:
    try:
        from normalizer import normalize_team
        return normalize_team(name)
    except Exception:
        return name.strip()


def _safe_int(v) -> Optional[int]:
    try: return int(float(v))
    except Exception: return None

def _safe_float(v) -> Optional[float]:
    try: return float(v)
    except Exception: return None


# ── Team IDs ──────────────────────────────────────────────────────────────────
_id_cache: Dict[str, Dict[str, str]] = {}   # league_code → {norm_name: team_id}

def _get_team_ids(league_code: str) -> Dict[str, str]:
    """
    Returns {normalised_name: espn_id} for all teams in league.
    Confirmed working: sports[0].leagues[0].teams[n].team.{id, displayName}
    """
    if league_code in _id_cache:
        return _id_cache[league_code]

    url  = ESPN_TEAMS.format(league=league_code)
    data = _get(url)
    if not data:
        _id_cache[league_code] = {}
        return {}

    mapping: Dict[str, str] = {}
    try:
        teams_list = (data["sports"][0]["leagues"][0]["teams"])
        for item in teams_list:
            t    = item.get("team", item)
            tid  = str(t.get("id", ""))
            name = t.get("displayName") or t.get("name", "")
            if tid and name:
                mapping[_norm(name)]  = tid
                mapping[name.strip()] = tid   # raw fallback
                if _DEBUG:
                    logger.debug(f"  ID map: {name} → {tid}")
    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Team ID parse failed [{league_code}]: {e}")

    logger.info(f"Team IDs [{league_code}]: {len(mapping)//2} teams")
    _id_cache[league_code] = mapping
    return mapping


# ── Team stats → xG proxy ─────────────────────────────────────────────────────
def _flatten_stats(data: dict) -> Dict[str, float]:
    """
    Flatten ESPN stats from ALL known response shapes into {name: value}.
    Handles: results[n].stats, splits.categories[n].stats, athletes[n].stats
    """
    flat: Dict[str, float] = {}

    def _ingest(stats_list):
        for s in stats_list:
            key = (s.get("name") or "").lower().replace(" ", "").replace("_", "")
            val = s.get("value")
            if key and val is not None:
                try:
                    flat[key] = float(val)
                except (ValueError, TypeError):
                    pass

    # Shape 1: data["results"][n]["stats"]
    for section in data.get("results", []):
        _ingest(section.get("stats", []))

    # Shape 2: data["splits"]["categories"][n]["stats"]
    for cat in data.get("splits", {}).get("categories", []):
        _ingest(cat.get("stats", []))

    # Shape 3: direct data["statistics"] (some leagues)
    _ingest(data.get("statistics", []))

    if _DEBUG and flat:
        logger.debug(f"  Stats keys: {list(flat.keys())[:25]}")

    return flat


def fetch_team_stats(team_id: str, league_code: str) -> Optional[Dict]:
    url  = ESPN_TEAM_STATS.format(league=league_code, team_id=team_id)
    data = _get(url)
    if not data:
        return None

    flat = _flatten_stats(data)
    if not flat:
        if _DEBUG:
            logger.debug(f"  No stats found. Top keys: {list(data.keys())}")
        return None

    # Games played — try several field names
    gp = _safe_float(
        flat.get("gamesplayed") or flat.get("matchesplayed") or
        flat.get("games") or flat.get("played")
    )
    if not gp:
        return None

    def _pg(keys):
        for k in keys:
            v = flat.get(k)
            if v is not None:
                return round(v / gp, 3)
        return None

    result: Dict = {"games_played": int(gp)}
    gf  = _pg(["goals", "goalsscored", "goalsfor"])
    sot = _pg(["shotsontarget", "ongoal", "targetsshots", "shotsontarget"])
    sht = _pg(["totalshots", "shots", "shotsattempted", "shotstaken"])
    ga  = _pg(["goalsagainst", "goalsallowed", "goalsconceded"])

    if gf  is not None: result["avg_goals_for"]      = gf
    if sht is not None: result["avg_shots"]          = sht
    if ga  is not None: result["avg_goals_against"]  = ga
    if sot is not None:
        result["avg_shots_on_target"] = sot
        result["avg_xg_for"]          = round(sot * 0.32, 3)

    return result if len(result) > 1 else None


# ── Rest days ─────────────────────────────────────────────────────────────────
def fetch_rest_days(team_id: str, league_code: str,
                    reference_date: Optional[str] = None) -> Optional[int]:
    url  = ESPN_SCHEDULE.format(league=league_code, team_id=team_id)
    data = _get(url)
    if not data:
        return None

    ref = datetime.now(timezone.utc)
    if reference_date:
        try:
            ref = datetime.fromisoformat(reference_date).replace(tzinfo=timezone.utc)
        except Exception:
            pass

    last_date = None
    for event in data.get("events", []):
        try:
            comp   = event.get("competitions", [{}])[0]
            status = comp.get("status", {}).get("type", {})
            if not status.get("completed", False):
                continue
            d = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
            if d < ref and (last_date is None or d > last_date):
                last_date = d
        except Exception:
            continue

    if not last_date:
        return None
    return max(int((ref - last_date).days), 1)


# ── Per-league enrichment ─────────────────────────────────────────────────────
def enrich_league(league_code: str, match_date: Optional[str] = None,
                  force: bool = False) -> int:
    league_name = ENRICH_LEAGUES.get(league_code)
    if not league_name:
        logger.warning(f"Unknown code: {league_code}")
        return 0

    logger.info(f"Enriching {league_name} ({league_code})…")

    team_ids = _get_team_ids(league_code)
    if not team_ids:
        logger.warning(f"  No team IDs for {league_code} — skipping")
        return 0

    # Which teams to process: fixtures today, else all teams in league
    if match_date:
        with _conn() as con:
            rows = con.execute(
                "SELECT home, away FROM fixtures WHERE match_date=? AND league=?",
                (match_date, league_name)
            ).fetchall()
        targets = list({t for r in rows for t in (r["home"], r["away"])})
        if not targets:
            logger.info(f"  No fixtures for {league_name} on {match_date}")
            return 0
    else:
        # Deduplicate: use normalised names from team_ids (every name appears twice: raw + norm)
        seen_ids = set()
        targets  = []
        for name, tid in team_ids.items():
            if tid not in seen_ids:
                seen_ids.add(tid)
                targets.append(name)

    enriched = 0
    for team in targets:
        # Skip if fresh cache (unless force)
        if not force:
            existing = get_cached_team_form(team, league_name)
            if existing and existing.get("avg_xg_for") is not None:
                logger.debug(f"  {team}: cache fresh, skipping")
                continue

        # Resolve team ID — exact match, then fuzzy 4-char prefix
        tid = team_ids.get(team) or team_ids.get(_norm(team))
        if not tid:
            for stored_name, stored_id in team_ids.items():
                if len(team) >= 4 and team[:4].lower() == stored_name[:4].lower():
                    tid = stored_id
                    break

        form_data: Dict = {}

        if tid:
            stats = fetch_team_stats(tid, league_code)
            if stats:
                form_data.update(stats)

            rest = fetch_rest_days(tid, league_code, match_date)
            if rest is not None:
                form_data["days_rest"] = rest
        else:
            logger.debug(f"  {team}: no ESPN ID found")

        if form_data:
            save_team_form(team, league_name, form_data, ttl_hours=12)
            enriched += 1
            logger.info(
                f"  ✓ {team}: "
                f"xG={form_data.get('avg_xg_for','—')}  "
                f"goals/g={form_data.get('avg_goals_for','—')}  "
                f"rest={form_data.get('days_rest','—')}d"
            )
        else:
            logger.debug(f"  {team}: no data retrieved")

    logger.info(f"Enriched {enriched}/{len(targets)} teams in {league_name}")
    return enriched


# ── Today's fixtures ──────────────────────────────────────────────────────────
def enrich_todays_fixtures(match_date: Optional[str] = None,
                            force: bool = False) -> Dict[str, int]:
    init_db()
    today = match_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with _conn() as con:
        rows = con.execute(
            "SELECT DISTINCT league FROM fixtures WHERE match_date=? AND status='unplayed'",
            (today,)
        ).fetchall()
    leagues_today = [r["league"] for r in rows]

    if not leagues_today:
        logger.info(f"No unplayed fixtures for {today}")
        return {}

    name_to_code = {v: k for k, v in ENRICH_LEAGUES.items()}
    results: Dict[str, int] = {}

    for league_name in leagues_today:
        code = name_to_code.get(league_name)
        if code:
            n = enrich_league(code, match_date=today, force=force)
            results[league_name] = n
        else:
            logger.debug(f"No ESPN code for: {league_name}")

    total = sum(results.values())
    logger.info(f"Enrichment complete: {total} teams across {len(results)} leagues")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-5s | %(name)-20s | %(message)s",
    )

    args = sys.argv[1:]

    if "--debug" in args:
        _DEBUG = True
        args = [a for a in args if a != "--debug"]

    force = False
    if "--refresh" in args:
        force = True
        args = [a for a in args if a != "--refresh"]
        init_db()
        with _conn() as con:
            n = con.execute("DELETE FROM team_form").rowcount
        logger.info(f"Cache cleared: {n} rows deleted")

    init_db()

    if not args:
        results = enrich_todays_fixtures(force=force)
        for lg, n in sorted(results.items()):
            print(f"  {lg:<30} {n} teams enriched")
    else:
        for code in args:
            enrich_league(code, force=force)

    with _conn() as con:
        n   = con.execute(
            "SELECT COUNT(*) as n FROM team_form WHERE expires_at > datetime('now')"
        ).fetchone()["n"]
        sample = con.execute(
            "SELECT team, league, avg_xg_for, days_rest FROM team_form "
            "WHERE expires_at > datetime('now') AND avg_xg_for IS NOT NULL LIMIT 5"
        ).fetchall()

    print(f"\nteam_form cache: {n} valid entries")
    if sample:
        print("Sample entries with xG:")
        for r in sample:
            print(f"  {r['team']:<30} {r['league']:<20} xG={r['avg_xg_for']}  rest={r['days_rest']}")

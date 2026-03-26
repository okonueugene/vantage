"""
understat_enricher.py — Vantage v4.1 Phase 2
=============================================
Fetches REAL xG data from Understat via Playwright.

One page load per league → xG for all teams in that league.
No Cloudflare, no API key, no per-team requests.

Coverage: EPL, La Liga, Bundesliga, Serie A, Ligue 1
  (5 page loads covers ~98 teams with real xG)

Data extracted per team:
  - avg_xg_for      (season xG per game — REAL, not proxy)
  - avg_xg_against  (defensive xG conceded per game)
  - avg_goals_for   (actual goals per game)
  - avg_goals_against
  - form_last5      (WDLWW from datesData)
  - form_xg_last5   (average xG in last 5 home/away games)
  - xg_source = "understat"

Usage:
    python understat_enricher.py                  # today's fixtures
    python understat_enricher.py EPL Bundesliga   # specific leagues
    python understat_enricher.py --refresh        # force re-scrape
"""

import json
import logging
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from playwright.sync_api import sync_playwright, Page, TimeoutError as PWTimeout

from store import init_db, save_team_form, get_cached_team_form, _conn

logger = logging.getLogger("Vantage.Understat")

SEASON = "2025"   # Understat uses start year

# Understat league codes → our league names
UNDERSTAT_LEAGUES = {
    "EPL":        "Premier League",
    "La_liga":    "La Liga",
    "Bundesliga": "Bundesliga",
    "Serie_A":    "Serie A",
    "Ligue_1":    "Ligue 1",
}

# Our league name → Understat code (reverse lookup)
_NAME_TO_CODE = {v: k for k, v in UNDERSTAT_LEAGUES.items()}

# Our league name → ESPN code (for rest-day fetching)
_LEAGUE_TO_ESPN = {
    "Premier League": "eng.1",
    "La Liga":        "esp.1",
    "Bundesliga":     "ger.1",
    "Serie A":        "ita.1",
    "Ligue 1":        "fra.1",
}

# Team name aliases: Understat name → our normalised name
# Only add entries where names differ
TEAM_ALIASES: Dict[str, str] = {
    "Manchester City":       "Man City",
    "Manchester United":     "Man Utd",
    "Tottenham":             "Spurs",
    "Newcastle United":      "Newcastle",
    "Nottingham Forest":     "Nottm Forest",
    "West Ham":              "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Leicester":             "Leicester City",
    "Brighton":              "Brighton",
    "Ipswich":               "Ipswich Town",
    "Atletico Madrid":       "Atletico Madrid",
    "Alaves":                "Alavés",
    "Celta Vigo":            "Celta Vigo",
    "Bayer Leverkusen":      "Bayer Leverkusen",
    "Borussia Dortmund":     "Dortmund",
    "Borussia M.Gladbach":   "Gladbach",
    "Eintracht Frankfurt":   "Frankfurt",
    "RasenBallsport Leipzig":"RB Leipzig",
    "Koln":                  "FC Cologne",
    "Inter":                 "Inter Milan",
    "AC Milan":              "AC Milan",
    "Hellas Verona":         "Verona",
    "Paris Saint Germain":   "PSG",
    "Saint-Etienne":         "Saint-Étienne",
    "Lens":                  "Lens",
}


def _norm_name(name: str) -> str:
    """Map Understat team name to our canonical name."""
    mapped = TEAM_ALIASES.get(name)
    if mapped:
        return mapped
    try:
        from normalizer import normalize_team
        return normalize_team(name)
    except Exception:
        return name.strip()


def _open_page() -> Tuple:
    """Start Playwright, return (playwright, browser, page)."""
    p = sync_playwright().start()
    browser = p.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage",
              "--disable-blink-features=AutomationControlled"]
    )
    ctx = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        locale="en-GB",
    )
    page = ctx.new_page()
    # Block ads/trackers so JS data scripts run faster
    page.route("**/*.{png,jpg,gif,svg,woff,woff2,ico}", lambda r: r.abort())
    for domain in ["adsbygoogle", "adlook", "googlesyndication",
                   "googletagmanager", "doubleclick", "adlook.tech"]:
        page.route(f"**/{domain}**", lambda r: r.abort())
    return p, browser, page


# ── Core scraper ──────────────────────────────────────────────────────────────
def scrape_league(league_code: str, page: Page,
                  season: str = SEASON) -> Dict[str, Dict]:
    """
    Load one Understat league page, extract teamsData + datesData.
    Returns {understat_team_name: {xg stats dict}}.
    One request covers all ~20 teams in the league.
    """
    url = f"https://understat.com/league/{league_code}/{season}"
    logger.info(f"  Loading {url}")
    time.sleep(random.uniform(2, 4))

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_function(
            "typeof teamsData !== 'undefined'", timeout=20000
        )
    except PWTimeout:
        logger.warning(f"  Timeout for {league_code}")
        return {}
    except Exception as e:
        logger.warning(f"  Failed to load {league_code}: {e}")
        return {}

    try:
        teams_raw  = page.evaluate("() => JSON.stringify(teamsData)")
        dates_raw  = page.evaluate("() => JSON.stringify(datesData)")
    except Exception as e:
        logger.warning(f"  evaluate() failed: {e}")
        return {}

    teams_data = json.loads(teams_raw)   # {id: {title, history:[{xG,xGA,scored,missed,...}]}}
    dates_data = json.loads(dates_raw)   # [{id, h:{title,xG}, a:{title,xG}, isResult, datetime}]

    logger.info(f"  {len(teams_data)} teams, {len(dates_data)} matches loaded")

    # Build per-team match history lookup from datesData
    # {team_title: [{xg_for, xg_ag, scored, missed, result, date, venue}]}
    team_matches: Dict[str, List[Dict]] = {}
    for match in dates_data:
        if not match.get("isResult"):
            continue
        try:
            h_title = match["h"]["title"]
            a_title = match["a"]["title"]
            h_xg    = float(match["xG"]["h"])
            a_xg    = float(match["xG"]["a"])
            h_goals = int(match["goals"]["h"])
            a_goals = int(match["goals"]["a"])
            dt      = match.get("datetime", "")

            for title, xg_for, xg_ag, scored, missed, venue in [
                (h_title, h_xg, a_xg, h_goals, a_goals, "Home"),
                (a_title, a_xg, h_xg, a_goals, h_goals, "Away"),
            ]:
                team_matches.setdefault(title, []).append({
                    "xg_for":  xg_for,
                    "xg_ag":   xg_ag,
                    "scored":  scored,
                    "missed":  missed,
                    "venue":   venue,
                    "result":  "W" if scored > missed else "D" if scored == missed else "L",
                    "date":    dt,
                })
        except (KeyError, ValueError, TypeError):
            continue

    # Build season stats per team
    results: Dict[str, Dict] = {}
    for team_id, team_obj in teams_data.items():
        title   = team_obj.get("title", "")
        history = team_obj.get("history", [])
        if not history:
            continue

        n = len(history)
        # history entries: {xG, xGA, scored, missed, wins, draws, loses, pts, ...}
        season_xg_for  = sum(float(h.get("xG",  0)) for h in history)
        season_xg_ag   = sum(float(h.get("xGA", 0)) for h in history)
        season_goals   = sum(int(h.get("scored", 0)) for h in history)
        season_concede = sum(int(h.get("missed", 0)) for h in history)

        avg_xg_for  = round(season_xg_for  / n, 3)
        avg_xg_ag   = round(season_xg_ag   / n, 3)
        avg_goals   = round(season_goals   / n, 3)
        avg_concede = round(season_concede / n, 3)

        # Form last 5 from datesData (more reliable order than history)
        matches_sorted = sorted(
            team_matches.get(title, []),
            key=lambda m: m["date"]
        )
        last5 = matches_sorted[-5:]
        form_str    = "".join(m["result"] for m in last5)
        form_xg     = round(sum(m["xg_for"] for m in last5) / max(len(last5), 1), 3)
        form_pts    = sum(3 if m["result"]=="W" else 1 if m["result"]=="D" else 0
                         for m in last5)

        # Home/away splits
        home_m = [m for m in matches_sorted if m["venue"] == "Home"]
        away_m = [m for m in matches_sorted if m["venue"] == "Away"]
        home_xg = round(sum(m["xg_for"] for m in home_m) / max(len(home_m), 1), 3)
        away_xg = round(sum(m["xg_for"] for m in away_m) / max(len(away_m), 1), 3)

        results[title] = {
            "games_played":      n,
            "avg_xg_for":        avg_xg_for,
            "avg_xga":           avg_xg_ag,       # store.py key
            "avg_xg_against":    avg_xg_ag,       # alias for sofascore compat
            "avg_goals_for":     avg_goals,
            "avg_goals_against": avg_concede,
            "form_last5":        form_str,
            "form_last5_points": form_pts,
            "form_xg_last5":     form_xg,
            "home_avg_xg":       home_xg,
            "away_avg_xg":       away_xg,
            "xg_source":         "understat",
        }

    return results


# ── Per-league enrichment ─────────────────────────────────────────────────────
def enrich_league(league_code: str, page: Page,
                  teams_filter: Optional[List[str]] = None,
                  force: bool = False) -> int:
    league_name = UNDERSTAT_LEAGUES.get(league_code)
    if not league_name:
        logger.warning(f"Unknown Understat code: {league_code}")
        return 0

    logger.info(f"Enriching {league_name} via Understat…")

    team_stats = scrape_league(league_code, page)
    if not team_stats:
        logger.warning(f"  No data returned for {league_code}")
        return 0

    enriched = 0
    for understat_name, stats in team_stats.items():
        our_name = _norm_name(understat_name)

        # If filter specified, only save matching teams
        if teams_filter:
            match = any(
                t.lower() in our_name.lower() or our_name.lower() in t.lower()
                for t in teams_filter
            )
            if not match:
                continue

        # Skip if any source already has xG (complement other enrichers; don't overwrite)
        if not force:
            existing = get_cached_team_form(our_name, league_name)
            if existing and existing.get("avg_xg_for") is not None:
                logger.debug(f"  {our_name}: xG already from {existing.get('xg_source','?')}, skip")
                continue

        # Fetch rest days from ESPN schedule (Understat has no schedule data)
        espn_code = _LEAGUE_TO_ESPN.get(league_name)
        if espn_code:
            try:
                from espn_enricher import _get_team_ids as espn_ids, fetch_rest_days
                id_map  = espn_ids(espn_code)
                espn_id = id_map.get(our_name) or id_map.get(understat_name)
                if espn_id:
                    rest = fetch_rest_days(espn_id, espn_code)
                    if rest is not None:
                        stats["days_rest"] = rest
            except Exception as e:
                logger.debug(f"  {our_name}: rest day fetch failed ({e})")

        save_team_form(our_name, league_name, stats, ttl_hours=36)
        enriched += 1
        logger.info(
            f"  ✓ {our_name}: "
            f"xG={stats['avg_xg_for']}  "
            f"xGA={stats['avg_xga']}  "
            f"form={stats['form_last5']}  "
            f"rest={stats.get('days_rest','—')}d"
        )

    logger.info(f"  Saved {enriched}/{len(team_stats)} teams for {league_name}")
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
    leagues_today = {r["league"] for r in rows}

    # Only process leagues Understat covers
    to_enrich = [
        (code, name) for code, name in UNDERSTAT_LEAGUES.items()
        if name in leagues_today
    ]

    if not to_enrich:
        logger.info("No Understat-covered leagues with fixtures today")
        return {}

    logger.info(
        f"Understat enrichment: {len(to_enrich)} leagues — "
        + ", ".join(name for _, name in to_enrich)
    )

    results: Dict[str, int] = {}
    p, browser, page = _open_page()
    try:
        for code, name in to_enrich:
            # Get today's teams for this league (for logging)
            with _conn() as con:
                rows = con.execute(
                    "SELECT home, away FROM fixtures WHERE match_date=? AND league=?",
                    (today, name)
                ).fetchall()
            today_teams = {t for r in rows for t in (r["home"], r["away"])}

            n = enrich_league(code, page, force=force)
            results[name] = n
    finally:
        browser.close()
        p.stop()

    total = sum(results.values())
    logger.info(f"Understat complete: {total} teams across {len(results)} leagues")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-5s | %(name)-20s | %(message)s",
    )

    args = sys.argv[1:]
    force = "--refresh" in args
    if force:
        args = [a for a in args if a != "--refresh"]

    init_db()

    if not args:
        results = enrich_todays_fixtures(force=force)
        for lg, n in sorted(results.items()):
            print(f"  {lg:<25} {n} teams enriched")
    else:
        # Treat args as Understat league codes: EPL, La_liga, Bundesliga, etc.
        p, browser, page = _open_page()
        try:
            for code in args:
                if code in UNDERSTAT_LEAGUES:
                    enrich_league(code, page, force=True)
                else:
                    print(f"Unknown code: {code}")
                    print(f"Valid codes: {list(UNDERSTAT_LEAGUES.keys())}")
        finally:
            browser.close()
            p.stop()

    # Summary
    with _conn() as con:
        rows = con.execute(
            "SELECT COUNT(*) as n, xg_source FROM team_form "
            "WHERE expires_at > datetime('now') AND avg_xg_for IS NOT NULL "
            "GROUP BY xg_source"
        ).fetchall()
    print(f"\nteam_form xG cache:")
    for r in rows:
        print(f"  source={r['xg_source'] or 'proxy':<12} count={r['n']}")

"""
sofascore_enricher.py — Vantage v4.1 Phase 2
=============================================
Fetches team stats from Sofascore via Playwright for leagues not covered
by Understat (Americas, Asia, Middle East).

Confirmed working (2026-03-07):
  Liga MX Clausura  → tid=11620  (Apertura=11621)
  Saudi Pro League  → tid=955
  J1 League         → tid=196
  Argentine Primera → tid=TBD (Primera Nacional 703 is 2nd div)

Data per team (from standings + team stats):
  goalsScored    → avg_goals_for
  goalsConceded  → avg_goals_against
  shotsOnTarget  → avg_shots_on_target → xG proxy (×0.32)
  bigChances     → big chance proxy
  wins/draws/losses → form
  scoresFor/Against → cross-check

Usage:
    python sofascore_enricher.py                   # today's fixtures
    python sofascore_enricher.py mex.1 usa.1       # specific league codes
    python sofascore_enricher.py --refresh         # force re-scrape
"""

import json
import logging
import re
import time
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from playwright.sync_api import sync_playwright, Page, TimeoutError as PWTimeout

from store import init_db, save_team_form, get_cached_team_form, _conn

logger = logging.getLogger("Vantage.Sofascore")

# ── Tournament registry ────────────────────────────────────────────────────────
# Maps our ESPN league code → (sofascore_tournament_id, our_league_name)
# Use Clausura for Liga MX (current running tournament)
# Fill in MLS/Brasileirao/Argentine once diagnose_sofascore4.py confirms IDs

SOFASCORE_LEAGUES: Dict[str, Tuple[int, str]] = {
    "mex.1":  (11620, "Liga MX"),
    "ksa.1":  (955,   "Saudi Pro League"),
    "jpn.1":  (196,   "J1 League"),
    "usa.1":  (242,   "MLS"),
    "chn.1":  (649,   "Chinese Super League"),
    "aus.1":  (136,   "A-League"),
    "den.1":  (39,    "Danish Superliga"),
    "sui.1":  (215,   "Swiss Super League"),
    "cze.1":  (172,   "Czech First League"),
    "rus.1":  (203,   "Russian Premier League"),
    "bra.1":  (325,   "Brasileirao"),
    "arg.1":  (155,   "Argentine Primera"),
    "aut.1":  (37,    "Austrian Bundesliga"),
}

# Our league name → ESPN code (reverse lookup)
_NAME_TO_CODE = {v[1]: k for k, v in SOFASCORE_LEAGUES.items()}

# Team name aliases: Sofascore name → our normalised name
TEAM_ALIASES: Dict[str, str] = {
    "Tigres UANL":              "Tigres Uanl",
    "Club América":             "América",
    "Deportivo Guadalajara":    "Guadalajara",
    "Atlético San Luis":        "Atlético De San Luis",
    "FC Juárez":                "Fc Juarez",
    "Mazatlán FC":              "Mazatlán Fc",
    "Pumas UNAM":               "Pumas Unam",
    "Cruz Azul":                "Cruz Azul",
    "CD Toluca":                "Toluca",
    "CF Monterrey":             "Monterrey",
    "Club Necaxa":              "Necaxa",
    "Club Santos Laguna":       "Santos",
    "Club Tijuana":             "Tijuana",
    "Club Puebla":              "Puebla",
    "Club León":                "León",
    "Club Atlas":               "Atlas",
    "Club Pachuca":             "Pachuca",
    "Club Querétaro":           "Querétaro",
    "Al Ahli":                  "Al-Ahli",
    "Al-Ahli":                  "Al-Ahli",
    # Argentine
    "Vélez Sársfield":          "Vélez Sársfield",
    "Atlético Tucumán":         "Atletico Tucuman",
    # Brazilian
    "Atlético Mineiro":         "Atletico Mineiro",
    "Vasco da Gama":            "Vasco Da Gama",
    "Grêmio":                   "Grêmio",
    "São Paulo":                "São Paulo",
    "RB Bragantino":            "Rb Bragantino",
    # MLS
    "Inter Miami CF":           "Inter Miami Cf",
    "LA Galaxy":                "La Galaxy",
    "LAFC":                     "Lafc",
    "New York City FC":         "New York City Fc",
    "Atlanta United FC":        "Atlanta United Fc",
    "Seattle Sounders FC":      "Seattle Sounders Fc",
    "FC Dallas":                "Fc Dallas",
    "Vancouver Whitecaps FC":   "Vancouver Whitecaps",
    "Austin FC":                "Austin Fc",
    "Charlotte FC":             "Charlotte Fc",
    "St. Louis City SC":        "St. Louis City Sc",
    "Minnesota United FC":      "Minnesota United Fc",
    "Nashville SC":             "Nashville Sc",
    "Orlando City SC":          "Orlando City Sc",
    "Chicago Fire FC":          "Chicago Fire Fc",
    "Philadelphia Union":       "Philadelphia Union",
    "San Diego FC":             "San Diego Fc",
}

def _norm(name: str) -> str:
    mapped = TEAM_ALIASES.get(name)
    if mapped:
        return mapped
    try:
        from normalizer import normalize_team
        return normalize_team(name)
    except Exception:
        return name.strip()


# ── Playwright helpers ────────────────────────────────────────────────────────
def _open_browser():
    p = sync_playwright().start()
    browser = p.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"]
    )
    page = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ).new_page()
    return p, browser, page


def _get_json(page: Page, url: str, retries: int = 2) -> Optional[Dict]:
    for attempt in range(retries + 1):
        try:
            time.sleep(random.uniform(0.8, 1.5))
            resp = page.goto(url, wait_until="domcontentloaded", timeout=20000)
            if resp.status == 404:
                return None
            if resp.status != 200:
                if attempt < retries:
                    time.sleep(2)
                    continue
                return None
            content = page.content()
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            if attempt == retries:
                logger.warning(f"  Failed [{url[:70]}]: {e}")
            else:
                time.sleep(2)
    return None


# ── Season cache ──────────────────────────────────────────────────────────────
_season_cache: Dict[int, int] = {}   # tid → latest season_id

def _get_latest_season(page: Page, tid: int) -> Optional[int]:
    if tid in _season_cache:
        return _season_cache[tid]
    data = _get_json(page, f"https://api.sofascore.com/api/v1/unique-tournament/{tid}/seasons")
    if not data:
        return None
    seasons = data.get("seasons", [])
    if not seasons:
        return None
    sid = seasons[0].get("id")
    _season_cache[tid] = sid
    logger.debug(f"  Season for tid={tid}: {seasons[0].get('name')} (id={sid})")
    return sid


# ── Core data fetch ───────────────────────────────────────────────────────────
def _fetch_league_data(page: Page, tid: int, sid: int
                       ) -> Tuple[Dict[str, Dict], Dict[str, int]]:
    """
    Returns:
      standings_map: {sofascore_name: {position, wins, draws, losses,
                                       goals_for, goals_against, points}}
      team_id_map:   {sofascore_name: sofascore_team_id}
    """
    url = (f"https://api.sofascore.com/api/v1/unique-tournament/{tid}"
           f"/season/{sid}/standings/total")
    data = _get_json(page, url)
    if not data:
        return {}, {}

    standings_map: Dict[str, Dict] = {}
    team_id_map:   Dict[str, int]  = {}

    for standing_group in data.get("standings", []):
        for row in standing_group.get("rows", []):
            try:
                team  = row.get("team", {})
                name  = team.get("name", "")
                tid2  = team.get("id")
                if not name or not tid2:
                    continue

                standings_map[name] = {
                    "position":      row.get("position"),
                    "wins":          row.get("wins", 0),
                    "draws":         row.get("draws", 0),
                    "losses":        row.get("losses", 0),
                    "goals_for":     row.get("scoresFor", 0),
                    "goals_against": row.get("scoresAgainst", 0),
                    "points":        row.get("points", 0),
                    "played":        (row.get("wins",0) + row.get("draws",0) +
                                      row.get("losses",0)),
                }
                team_id_map[name] = tid2
            except Exception:
                continue

    logger.debug(f"  Standings: {len(standings_map)} teams")
    return standings_map, team_id_map


def _fetch_team_stats(page: Page, team_id: int, tid: int, sid: int) -> Optional[Dict]:
    """
    Fetch per-team statistics from Sofascore.
    Key fields: goalsScored, goalsConceded, shotsOnTarget, bigChances, matches
    """
    url = (f"https://api.sofascore.com/api/v1/team/{team_id}"
           f"/unique-tournament/{tid}/season/{sid}/statistics/overall")
    data = _get_json(page, url)
    if not data:
        return None

    stats = data.get("statistics", {})
    if not stats:
        return None

    matches = stats.get("matches") or stats.get("gamesPlayed") or 1
    if matches == 0:
        return None

    def _pg(key, fallback=None):
        v = stats.get(key, fallback)
        return round(float(v) / matches, 3) if v is not None else None

    gf  = _pg("goalsScored")
    ga  = _pg("goalsConceded")
    sot = _pg("shotsOnTarget")
    bc  = _pg("bigChances")
    bc_created = _pg("bigChancesCreated")

    result: Dict = {"games_played": int(matches)}
    if gf  is not None: result["avg_goals_for"]      = gf
    if ga  is not None: result["avg_goals_against"]  = ga
    if sot is not None:
        result["avg_shots_on_target"] = sot
        result["avg_xg_for"]          = round(sot * 0.32, 3)
    if bc  is not None: result["avg_big_chances"]    = bc

    return result if len(result) > 1 else None


# ── Per-league enrichment ─────────────────────────────────────────────────────
def enrich_league(league_code: str, page: Page, force: bool = False,
                  match_date: Optional[str] = None) -> int:
    entry = SOFASCORE_LEAGUES.get(league_code)
    if not entry:
        logger.warning(f"No Sofascore config for: {league_code}")
        return 0

    tid, league_name = entry
    logger.info(f"Enriching {league_name} via Sofascore (tid={tid})…")

    sid = _get_latest_season(page, tid)
    if not sid:
        logger.warning(f"  No season found for {league_name}")
        return 0

    standings, team_ids = _fetch_league_data(page, tid, sid)
    if not standings:
        logger.warning(f"  No standings for {league_name}")
        return 0

    # Which teams to enrich: fixture teams for match_date (for logging "TODAY")
    date_str = match_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _conn() as con:
        rows = con.execute(
            "SELECT home, away FROM fixtures WHERE match_date=? AND league=?",
            (date_str, league_name)
        ).fetchall()
    fixture_teams = {t for r in rows for t in (r["home"], r["away"])}

    enriched = 0
    for ss_name, standing in standings.items():
        our_name = _norm(ss_name)

        # Skip if any source already has xG (complement other enrichers; don't overwrite)
        if not force:
            existing = get_cached_team_form(our_name, league_name)
            if existing and existing.get("avg_xg_for") is not None:
                logger.debug(f"  {our_name}: xG already from {existing.get('xg_source','?')}, skip")
                continue

        team_id = team_ids.get(ss_name)
        form_data: Dict = {
            "position":          standing.get("position"),
            "xg_source":         "goals_proxy",   # default — upgraded below if real xG fetched
        }

        # Derive form string from W/D/L record (rough — no sequence)
        played = standing.get("played", 0)
        if played > 0:
            w = standing.get("wins", 0)
            d = standing.get("draws", 0)
            gf = standing.get("goals_for", 0)
            ga = standing.get("goals_against", 0)
            form_data["avg_goals_for"]     = round(gf / played, 3)
            form_data["avg_goals_against"] = round(ga / played, 3)

        if team_id:
            stats = _fetch_team_stats(page, team_id, tid, sid)
            if stats:
                form_data.update(stats)
                # Upgrade source tag: sofascore = real shot-quality xG from stats endpoint
                # goals_proxy = only goals/g from standings (no shot data available)
                if form_data.get("avg_xg_for") is not None:
                    form_data["xg_source"] = "sofascore"

        # Only save if we have something meaningful
        if form_data.get("avg_xg_for") or form_data.get("avg_goals_for"):
            # Fetch rest days from ESPN schedule (Sofascore has no schedule API)
            try:
                from espn_enricher import _get_team_ids as espn_ids, fetch_rest_days
                id_map = espn_ids(league_code)
                espn_tid = id_map.get(our_name) or id_map.get(ss_name)
                if espn_tid:
                    rest = fetch_rest_days(espn_tid, league_code)
                    if rest is not None:
                        form_data["days_rest"] = rest
            except Exception:
                pass

            save_team_form(our_name, league_name, form_data, ttl_hours=36)
            enriched += 1
            logger.info(
                f"  ✓ {our_name}: "
                f"xG={form_data.get('avg_xg_for','—')}  "
                f"goals/g={form_data.get('avg_goals_for','—')}  "
                f"pos={form_data.get('position','—')}  "
                f"rest={form_data.get('days_rest','—')}d"
                + (" ← TODAY" if our_name in fixture_teams else "")
            )
        else:
            logger.debug(f"  {our_name}: no usable stats")

    logger.info(f"Enriched {enriched}/{len(standings)} teams in {league_name}")
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

    to_enrich = [
        code for code, (_, name) in SOFASCORE_LEAGUES.items()
        if name in leagues_today
    ]

    if not to_enrich:
        logger.info("No Sofascore-covered leagues with fixtures today")
        return {}

    names = [SOFASCORE_LEAGUES[c][1] for c in to_enrich]
    logger.info(f"Sofascore enrichment: {', '.join(names)}")

    results: Dict[str, int] = {}
    p, browser, page = _open_browser()
    try:
        for code in to_enrich:
            _, league_name = SOFASCORE_LEAGUES[code]
            n = enrich_league(code, page, force=force, match_date=today)
            results[league_name] = n
    finally:
        browser.close()
        p.stop()

    return results


def add_league(espn_code: str, sofascore_tid: int, league_name: str):
    """Runtime helper to add a new league without editing source."""
    SOFASCORE_LEAGUES[espn_code] = (sofascore_tid, league_name)
    _NAME_TO_CODE[league_name] = espn_code
    logger.info(f"Added: {league_name} ({espn_code}) → Sofascore tid={sofascore_tid}")


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
        p, browser, page = _open_browser()
        try:
            for code in args:
                if code in SOFASCORE_LEAGUES:
                    enrich_league(code, page, force=True)
                else:
                    print(f"Unknown code: {code}")
                    print(f"Available: {list(SOFASCORE_LEAGUES.keys())}")
        finally:
            browser.close()
            p.stop()

    with _conn() as con:
        rows = con.execute(
            "SELECT COUNT(*) as n, xg_source FROM team_form "
            "WHERE expires_at > datetime('now') AND avg_xg_for IS NOT NULL "
            "GROUP BY xg_source"
        ).fetchall()
    print(f"\nxG cache by source:")
    for r in rows:
        print(f"  {r['xg_source'] or 'proxy':<15} {r['n']} teams")

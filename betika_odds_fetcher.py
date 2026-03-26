import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
import time
import json
from dateutil import parser, tz
from fuzzywuzzy import fuzz

# CONFIG ────────────────────────────────────────────────────────────────
FUZZY_TEAM_THRESHOLD   = 78
TIME_TOLERANCE_HOURS   = 5.0
MIN_ACCEPTABLE_SCORE   = 100
BETIKA_TIMEOUT         = 6      # seconds — fail fast, don't block pipeline
BETIKA_RETRIES         = 1      # one retry on timeout before giving up

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Betika start_time is EAT (Africa/Nairobi = UTC+3), naive string.
# Always interpret with this TZ for correct UTC comparison.
BETIKA_TZ = tz.gettz("Africa/Nairobi")   # UTC+3, no DST

# Leagues Betika actually carries. Searches for other leagues are skipped
# immediately — no network call, no timeout waste.
# Key = Vantage league name (from ESPN normalisation).
BETIKA_SUPPORTED_LEAGUES = {
    # Europe top 5
    "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
    # Europe mid-tier
    "Eredivisie", "Belgian Pro League", "Primeira Liga", "Süper Lig",
    "Scottish Premiership", "Championship", "League One", "League Two",
    "2. Bundesliga", "Serie B", "Ligue 2", "Segunda Division",
    "Austrian Bundesliga", "Czech First League", "Swiss Super League",
    "Danish Superliga", "Russian Premier League", "Super League Greece",
    # Americas — Betika carries top South American but Argentine/Brasileirao
    # are rarely listed; MLS and Liga MX are more reliably covered.
    "MLS", "Liga MX",
    # Africa / Middle East (common on Betika Kenya)
    "Saudi Pro League",
    # European cups
    "Champions League", "Europa League", "Conference League",
    "FA Cup", "Coppa Italia", "Copa del Rey", "DFB-Pokal", "Coupe de France",
    # International — Betika Kenya is very strong on international fixtures,
    # especially WCQ and Nations League during international windows.
    "UEFA World Cup Qualifiers",
    "CONMEBOL World Cup Qualifiers",
    "CONCACAF World Cup Qualifiers",
    "CAF World Cup Qualifiers",
    "AFC World Cup Qualifiers",
    "OFC World Cup Qualifiers",
    "UEFA Nations League",
    "CONCACAF Nations League",
    "Copa America",
    "AFCON",
    "AFC Asian Cup",
    "International Friendlies",
}


class BetikaOddsFetcher:
    BASE_SEARCH = "https://api.betika.com/v1/uo/matches"
    BASE_MATCH  = "https://api.betika.com/v1/uo/match"
    ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })

    def fetch_espn_fixtures(self, league: str, date: str = None) -> List[Dict]:
        if not date:
            date = datetime.now().strftime("%Y%m%d")
        params = {"dates": date}
        try:
            url = self.ESPN_SCOREBOARD.format(league=league)
            r = self.session.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            fixtures = []
            for event in data.get("events", []):
                comps = event.get("competitions", [{}])[0]
                fixture = {
                    "espn_id": event.get("id"),
                    "home_team": comps.get("competitors", [{}])[0].get("team", {}).get("displayName", ""),
                    "away_team": comps.get("competitors", [{}])[1].get("team", {}).get("displayName", ""),
                    "date": event.get("date"),
                    "competition": event.get("league", {}).get("name", "")
                }
                fixtures.append(fixture)
            logger.info(f"Fetched {len(fixtures)} fixtures from ESPN for {league} on {date}")
            return fixtures
        except Exception as e:
            logger.error(f"Failed to fetch ESPN fixtures for {league}: {e}")
            return []

    def normalize_team(self, name: str) -> str:
        if not name: return ""
        name = name.lower().strip()
        name = name.replace("fc ", "").replace("ac ", "").replace("rc ", "").replace("sc ", "")
        name = name.replace("u19", "").replace("women", "").replace("féminin", "").replace(" ", "").replace("-", "")
        name = name.translate(str.maketrans("éèêëàâäîïôöùûüç", "eeeeaaaiioouuuc"))
        return name

    def teams_match(self, espn_h: str, espn_a: str, b_h: str, b_a: str) -> bool:
        nh, na = self.normalize_team(espn_h), self.normalize_team(espn_a)
        bh, ba = self.normalize_team(b_h), self.normalize_team(b_a)
        best = max(
            fuzz.ratio(nh, bh) + fuzz.ratio(na, ba),
            fuzz.ratio(nh, ba) + fuzz.ratio(na, bh)
        )
        return best >= FUZZY_TEAM_THRESHOLD * 2

    def search_betika(self, keyword: str, limit: int = 30) -> List[Dict]:
        params = {
            "page": 1,
            "limit": limit,
            "keyword": keyword.strip(),
            "tab": '""',
            "sub_type_id": "1,186"
        }
        for attempt in range(BETIKA_RETRIES + 1):
            try:
                r = self.session.get(self.BASE_SEARCH, params=params,
                                     timeout=BETIKA_TIMEOUT)
                r.raise_for_status()
                return r.json().get("data", [])
            except requests.exceptions.Timeout:
                if attempt < BETIKA_RETRIES:
                    time.sleep(1)
                    continue
                logger.warning(f"Betika search timed out '{keyword}' (gave up after {BETIKA_RETRIES+1} attempts)")
                return []
            except Exception as e:
                logger.warning(f"Betika search failed '{keyword}': {e}")
                return []
        return []

    def get_full_odds(self, parent_match_id: str) -> List[Dict]:
        try:
            r = self.session.get(self.BASE_MATCH, params={"parent_match_id": parent_match_id}, timeout=12)
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception as e:
            logger.warning(f"get_full_odds failed {parent_match_id}: {e}")
            return []

    def extract_odds_dict(self, markets: List[Dict]) -> Dict:
        """
        Extract only reliable markets:
        - 1X2
        - BTTS
        - Over/Under goals >= 4.5 only (lines 0.5–3.5 are corrupted on Betika API)
        - Corners totals (all lines reliable)
        - Cards totals (all lines reliable)
        """
        result = {}

        for m in markets:
            sub_id = m.get("sub_type_id")
            name = m.get("name", "").lower()

            # 1X2
            if sub_id == "1" or "1x2" in name:
                for o in m.get("odds", []):
                    key = o.get("display")
                    if key in ("1", "X", "2"):
                        result[key] = o["odd_value"]

            # BTTS
            if sub_id == "29" or "both teams to score" in name or "gg/ng" in name:
                for o in m.get("odds", []):
                    disp = o.get("display", "").upper()
                    if disp == "YES":
                        result["btts_yes"] = o["odd_value"]
                    elif disp == "NO":
                        result["btts_no"] = o["odd_value"]

            # Goals O/U — skip lines < 4.5 (API bug)
            if sub_id == "18" or ("total" in name and "corner" not in name and "card" not in name and "booking" not in name):
                for o in m.get("odds", []):
                    sbv = o.get("special_bet_value", "")
                    if "total=" not in sbv: continue
                    try:
                        line = float(sbv.split("total=")[-1])
                        if line < 4.5: continue   # ← corrupted on Betika
                    except:
                        continue
                    disp = o.get("display", "").upper()
                    key_prefix = "over" if "OVER" in disp else ("under" if "UNDER" in disp else None)
                    if key_prefix:
                        result[f"{key_prefix}{int(line*10):02d}"] = o["odd_value"]

            # Corners
            if "corner" in name and "total" in name:
                for o in m.get("odds", []):
                    sbv = o.get("special_bet_value", "")
                    if "total=" not in sbv: continue
                    try:
                        line = float(sbv.split("total=")[-1])
                    except:
                        continue
                    disp = o.get("display", "").upper()
                    key_prefix = "corners_over" if "OVER" in disp else ("corners_under" if "UNDER" in disp else None)
                    if key_prefix:
                        result[f"{key_prefix}{int(line*10):02d}"] = o["odd_value"]

            # Cards / bookings
            if any(k in name for k in ["card", "booking", "yellow", "red"]) and "total" in name:
                for o in m.get("odds", []):
                    sbv = o.get("special_bet_value", "")
                    if "total=" not in sbv: continue
                    try:
                        line = float(sbv.split("total=")[-1])
                    except:
                        continue
                    disp = o.get("display", "").upper()
                    key_prefix = "cards_over" if "OVER" in disp else ("cards_under" if "UNDER" in disp else None)
                    if key_prefix:
                        result[f"{key_prefix}{int(line*10):02d}"] = o["odd_value"]

        # ── Draw odds sanity check ────────────────────────────────────────────
        # Betika inflates X odds for heavy-favourite home games.
        # E.g. Milan vs Torino: Betika X=11.0, true market X=4.5.
        # Rule: if home_win < 1.40 (heavy home favourite), X should be ≤ 7.0.
        # If X > max_expected_draw, discard it — APIF/TheOddsAPI has the correct line.
        h = result.get("1")
        x = result.get("X")
        a = result.get("2")
        if h and x and a:
            try:
                h, x, a = float(h), float(x), float(a)
                # Estimate fair draw odds from Shin-adjusted 1X2
                p1 = 1/h; px = 1/x; p2 = 1/a
                vig = p1 + px + p2
                p1_fair = p1/vig; px_fair = px/vig
                fair_x = 1/px_fair if px_fair > 0 else 999
                # If Betika's X is more than 2.2× the fair value, it's corrupted
                if x > fair_x * 2.2:
                    logger.debug(f"Betika draw odds X={x} discarded (fair={fair_x:.1f}, ratio={x/fair_x:.1f}×)")
                    del result["X"]
                else:
                    result["1"] = h
                    result["X"] = x
                    result["2"] = a
            except Exception:
                pass

        return result

    def normalize_to_utc(self, dt_str: str, is_betika: bool = False) -> datetime:
        """
        Parse a datetime string to UTC.
        - ESPN dates are already UTC (ISO 8601 with Z).
        - Betika start_time is a naive string in EAT (Africa/Nairobi, UTC+3, no DST).
          is_betika=True forces EAT interpretation.
        """
        dt = parser.parse(dt_str)
        if is_betika or dt.tzinfo is None:
            # Treat as EAT (UTC+3)
            dt = dt.replace(tzinfo=BETIKA_TZ)
        return dt.astimezone(tz.UTC)

    def find_and_get_odds(self, espn_fixture: Dict) -> Optional[Dict]:
        home = espn_fixture.get("home_team", "")
        away = espn_fixture.get("away_team", "")
        date_str = espn_fixture.get("date")
        espn_id = espn_fixture.get("espn_id")

        if not date_str:
            logger.warning(f"{espn_id} missing date")
            return None

        try:
            espn_utc = self.normalize_to_utc(date_str, is_betika=False)
        except Exception as e:
            logger.warning(f"Parse ESPN date failed {date_str}: {e}")
            return None

        # Search by home team, fall back to away
        candidates = self.search_betika(home)
        if not candidates:
            candidates = self.search_betika(away)
        if not candidates:
            logger.warning(f"No candidates for {home} vs {away}")
            return None

        best, best_score, best_diff = None, -1, float('inf')
        betika_utc_final = None

        for c in candidates:
            c_h = c.get("home_team", "")
            c_a = c.get("away_team", "")
            c_time = c.get("start_time")

            # ── No competition filter: team fuzzy match is the guard ──────
            # (removing "ligue only" filter so this works for all leagues)

            # Time diff — interpret Betika time as EAT
            time_ok, diff_h = False, float('inf')
            betika_utc = None
            if c_time:
                try:
                    betika_utc = self.normalize_to_utc(c_time, is_betika=True)
                    diff_h = abs((espn_utc - betika_utc).total_seconds()) / 3600
                    time_ok = diff_h <= TIME_TOLERANCE_HOURS
                except:
                    pass

            fuzzy_ok = self.teams_match(home, away, c_h, c_a)

            score = 0
            if fuzzy_ok: score += 100
            if time_ok:  score += 50 - diff_h * 2

            if score > best_score:
                best_score = score
                best = c
                best_diff = diff_h
                betika_utc_final = betika_utc

        if best and best_score >= MIN_ACCEPTABLE_SCORE:
            pid = best.get("parent_match_id")
            if pid:
                markets = self.get_full_odds(pid)
                odds = self.extract_odds_dict(markets)
                logger.info(
                    f"Betika matched: {home} vs {away} | "
                    f"score={best_score} diff={best_diff:.1f}h | "
                    f"1={odds.get('1','?')} X={odds.get('X','?')} 2={odds.get('2','?')}"
                )
                return {
                    "espn_id": espn_id,
                    "espn_home": home,
                    "espn_away": away,
                    "espn_utc": str(espn_utc),
                    "betika_home": best.get("home_team"),
                    "betika_away": best.get("away_team"),
                    "betika_time": best.get("start_time"),
                    "betika_utc": str(betika_utc_final) if betika_utc_final else "N/A",
                    "competition": best.get("competition_name"),
                    "odds": odds,
                    "score": best_score,
                    "time_diff_hours": round(best_diff, 1),
                    "source": "Betika"
                }

        logger.warning(f"No good match {home} vs {away} | best score {best_score}")
        return None

    # ── Integration entry point ───────────────────────────────────────────────
    def enrich(self, fixtures: List[Dict]) -> int:
        """
        Enrich a list of Vantage fixture dicts in-place.
        Writes to fixture["odds"]: 1X2, BTTS, corners, cards.
        Does NOT overwrite over25/under25 — those come from TheOddsAPI
        (Betika goals lines 0.5–3.5 are corrupted).
        Returns count enriched.
        """
        enriched = 0
        skipped_leagues = set()
        for fx in fixtures:
            league = fx.get("league", "")
            if league not in BETIKA_SUPPORTED_LEAGUES:
                skipped_leagues.add(league)
                continue

            espn_fixture = {
                "home_team": fx.get("home", ""),
                "away_team": fx.get("away", ""),
                "date":      fx.get("kickoff_utc", ""),
                "espn_id":   fx.get("canonical_id", ""),
            }
            result = self.find_and_get_odds(espn_fixture)
            if not result:
                continue

            odds = result.get("odds", {})
            fx_odds = fx.setdefault("odds", {})

            for key, val in odds.items():
                # 1X2 → home_win / draw / away_win keys used by Vantage
                if key == "1":
                    fx_odds["home_win"] = float(val)
                elif key == "X":
                    fx_odds["draw"] = float(val)
                elif key == "2":
                    fx_odds["away_win"] = float(val)
                # BTTS — write directly
                elif key in ("btts_yes", "btts_no"):
                    try:
                        fx_odds[key] = float(val)
                    except (ValueError, TypeError):
                        pass
                # Corners — write all granular lines, also set odds_corners_over
                # to the preferred model line (9.5) so predictor can use it.
                elif key.startswith("corners_"):
                    try:
                        fx_odds[key] = float(val)
                        # Extract line from key e.g. corners_over95 → 9.5
                        if key.startswith("corners_over") and key[12:].isdigit():
                            line = int(key[12:]) / 10.0
                            # Prefer 9.5; if not available, take first over line
                            if line == 9.5:
                                fx_odds["odds_corners_over"] = float(val)
                                fx_odds["corners_line"] = 9.5
                            elif "odds_corners_over" not in fx_odds:
                                fx_odds["odds_corners_over"] = float(val)
                                fx_odds["corners_line"] = line
                    except (ValueError, TypeError):
                        pass
                # Cards — write all granular lines
                elif key.startswith("cards_"):
                    try:
                        fx_odds[key] = float(val)
                        if key.startswith("cards_over") and key[10:].isdigit():
                            line = int(key[10:]) / 10.0
                            if line == 3.5:
                                fx_odds["odds_cards_over"] = float(val)
                                fx_odds["cards_line"] = 3.5
                            elif "odds_cards_over" not in fx_odds:
                                fx_odds["odds_cards_over"] = float(val)
                                fx_odds["cards_line"] = line
                    except (ValueError, TypeError):
                        pass
                # Goals O/U 4.5+ — only fill if TheOddsAPI didn't already set over25
                elif (key.startswith("over") or key.startswith("under")) and \
                     "over25" not in fx_odds:
                    try:
                        fx_odds[key] = float(val)
                    except (ValueError, TypeError):
                        pass

            fx["betika_matched"] = True
            fx["_odds_fresh"] = True
            enriched += 1

        if skipped_leagues:
            logger.debug(f"Betika: skipped unsupported leagues: {', '.join(sorted(skipped_leagues))}")
        logger.info(f"Betika enrichment complete — {enriched}/{len(fixtures)} fixtures enriched")
        return enriched


# ── Standalone test ───────────────────────────────────────────────────────────
def main():
    fetcher = BetikaOddsFetcher()
    date_str = "20260308"

    # Test across multiple leagues
    test_leagues = {
        "fra.1": "Ligue 1",
        "ger.1": "Bundesliga",
        "ita.1": "Serie A",
        "esp.1": "La Liga",
    }

    all_results = []
    for league_code, league_name in test_leagues.items():
        logger.info(f"\n{'='*60}\nFetching {league_name}...\n{'='*60}")
        fixtures = fetcher.fetch_espn_fixtures(league_code, date_str)
        if not fixtures:
            continue
        for fix in fixtures:
            res = fetcher.find_and_get_odds(fix)
            if res:
                all_results.append(res)
            time.sleep(1.5)

    print(f"\n\nTotal matched: {len(all_results)}")
    for r in all_results:
        print(f"\n{r['espn_home']} vs {r['espn_away']} ({r['competition']})")
        print(f"  1X2: {r['odds'].get('1','?')} / {r['odds'].get('X','?')} / {r['odds'].get('2','?')}")
        print(f"  BTTS: yes={r['odds'].get('btts_yes','?')} no={r['odds'].get('btts_no','?')}")
        print(f"  Corners 9.5: over={r['odds'].get('corners_over95','?')} under={r['odds'].get('corners_under95','?')}")
        print(f"  Cards 2.5: over={r['odds'].get('cards_over25','?')} under={r['odds'].get('cards_under25','?')}")
        print(f"  Time diff: {r['time_diff_hours']}h | Score: {r['score']}")


if __name__ == "__main__":
    main()

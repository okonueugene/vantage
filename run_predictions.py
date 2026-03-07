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
    get_hit_rate_by_market, storage_summary, settle_result,
    stale_fixture_guard, _conn,
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
}

def _normalize_league(name: str) -> str:
    """Resolve ESPN league name variants to our canonical names."""
    return _LEAGUE_ALIASES.get(name, name)


# ══════════════════════════════════════════════
# STEP 1 — Fetch or load from store
# ══════════════════════════════════════════════
def fetch_or_load_fixtures(
    target_date: Optional[str] = None,
    lookahead: int = 2,
) -> Tuple[List[Dict], bool]:
    """
    Load fixtures for target_date (default: today UTC).
    If no fixtures found, look ahead up to `lookahead` days.
    Always runs stale_fixture_guard to drop matches that have already kicked off.
    """
    today = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Try cache first for the target date
    cached = get_todays_fixtures(today)
    if cached:
        logger.info(f"Store hit: {len(cached)} fixtures cached for {today}")
        for fx in cached:
            if "odds" not in fx or not fx["odds"]:
                fx["odds"] = {}
                for m in ["home_win","draw","away_win","over25","under25"]:
                    v = _get_stored_odds(fx.get("canonical_id",""), m)
                    if v:
                        fx["odds"][m] = v

        fixtures, stale = stale_fixture_guard(cached)
        _log_stale(stale)
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

    # Only use real form data — do NOT fall back to league prior as fake xG.
    # If no form data exists, leave xG as None so predictor knows it has no team data.
    home_xg = home_form.get("avg_xg_for")   # None if no cached form
    away_xg = away_form.get("avg_xg_for")   # None if no cached form

    ref_name = fixture.get("referee","")
    drs_base = get_ref_red_card_rate(ref_name, league) if ref_name else 0.12
    inj_bump = ((home_form.get("injury_count",0) or 0) + (away_form.get("injury_count",0) or 0)) * 0.005
    drs      = round(min(drs_base + inj_bump, 0.40), 3)

    home_pos = home_form.get("league_position", 10) or 10
    away_pos = away_form.get("league_position", 10) or 10
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
                for market in ("home_win", "draw", "away_win", "over25", "under25"):
                    if market in raw_odds and market not in odds:
                        odds[market] = raw_odds[market]
            except Exception:
                pass
    for market in ["home_win","draw","away_win","over25","under25"]:
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
    logger.info(f"Blend weight: {blend_weight:.3f}  (n={brier.get('n',0)} calibration bets)")

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
            # xG features (for Poisson lambdas + GameTempo)
            "home_xg":         enriched.get("home_xg"),
            "away_xg":         enriched.get("away_xg"),
            "home_n_matches":  enriched.get("home_n_matches", 5),
            # Corners features
            "home_corners_avg": enriched.get("home_corners_avg"),
            "away_corners_avg": enriched.get("away_corners_avg"),
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
            raw_stake = min(raw_kelly*0.5, 0.03) * bankroll
            adj = risk.adjust_stake(raw_stake, signal)

            drift = odds_map.get(f"{pred.market}_drift_pct")
            drift_note = ""
            if drift and drift > 5:
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
        cached_xg = con.execute(
            "SELECT COUNT(DISTINCT team) as n FROM team_form "
            "WHERE expires_at > ? AND avg_xg_for IS NOT NULL",
            (now.isoformat(),)
        ).fetchone()["n"] or 0
    cache_pct = cached_xg / max(total_fx, 1)

    if cache_pct >= 0.20:
        logger.info(f"Form cache warm ({cached_xg}/{total_fx} teams with xG) — skipping enrichment")
    else:
        logger.info(f"Form cache cold ({cached_xg}/{total_fx} teams) — running enrichers")

        # Stage 1: FBref real xG (top leagues only)
        try:
            from fbref_enricher import enrich_todays_fixtures as fbref_enrich
            fbref_enrich(match_date=today)
            logger.info("FBref enrichment complete")
        except Exception as e:
            logger.warning(f"FBref enricher failed (non-fatal): {e}")

        # Stage 2: ESPN fallback for remaining teams
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
            "blend_weight":        edge_bets[0]["blend_weight"] if edge_bets else 0.6,
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
    from store import _conn
    from calibration import CalibrationLab
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
    if brier.get("n"):
        print(f"\n  BRIER SCORES (n={brier['n']})")
        print(f"  Model: {brier['model_brier']:.5f}  Market: {brier['market_brier']:.5f}")
        print(f"  → Blend weight: {brier['recommended_blend']:.3f}")

    markets = get_hit_rate_by_market()
    if markets:
        print(f"\n  BY MARKET")
        for m in markets:
            print(f"  {m['market']:<12} n={m['n']:>4}  hit={m['hit_rate']*100:.1f}%  roi={((m['roi'] or 0)*100):.1f}%")

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

    sub.add_parser("stats")

    args = parser.parse_args()
    if args.cmd == "settle":
        cmd_settle(args)
    elif args.cmd == "stats":
        cmd_stats()
    else:
        run(bankroll=args.bankroll, min_ev=args.min_ev,
            target_date=args.date, lookahead=args.lookahead)

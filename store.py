"""
store.py
────────
Persistent local SQLite storage for the Vantage Engine.

Tables:
  fixtures      — every fetched fixture (deduplicated by canonical_id + date)
  odds          — odds snapshots per fixture (tracks line movement)
  predictions   — model output per fixture+market
  results       — settled outcomes for CLV + Brier score tracking
  league_stats  — rolling league xG/goals means for regime detection
  ref_stats     — referee red card rates for DRS
  team_form     — team form/xG/position cache (TTL: 24h)

Benefits:
  - No re-fetching fixtures already seen today
  - Historical regime detection from real league goal means
  - Line movement tracking (opening vs closing odds)
  - Brier score calibration from settled results
  - Zero API calls for already-cached team data
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Vantage.Store")

DB_PATH = Path("vantage_store.db")


# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS fixtures (
    canonical_id    TEXT PRIMARY KEY,
    home            TEXT NOT NULL,
    away            TEXT NOT NULL,
    league          TEXT NOT NULL,
    kickoff_utc     TEXT,
    match_date      TEXT NOT NULL,   -- YYYY-MM-DD
    status          TEXT DEFAULT 'unplayed',  -- unplayed / live / finished
    source          TEXT,
    fetched_at      TEXT NOT NULL,
    raw_json        TEXT             -- full source record
);

CREATE TABLE IF NOT EXISTS odds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id    TEXT NOT NULL,
    snapshot_time   TEXT NOT NULL,
    market          TEXT NOT NULL,   -- home_win / draw / away_win / over25 / under25
    odds_decimal    REAL NOT NULL,
    source          TEXT,
    is_opening      INTEGER DEFAULT 0,  -- 1 = first snapshot of day
    is_closing      INTEGER DEFAULT 0,  -- 1 = last snapshot before kickoff
    FOREIGN KEY (canonical_id) REFERENCES fixtures(canonical_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id    TEXT NOT NULL,
    run_id          TEXT NOT NULL,   -- scan_id from engine run
    generated_at    TEXT NOT NULL,
    market          TEXT NOT NULL,
    odds            REAL,
    p_true          REAL,
    p_market        REAL,
    p_model         REAL,
    ev_pct          REAL,
    regime          TEXT,
    drs             REAL,
    motivation_delta INTEGER,
    blend_weight    REAL,
    stake_pct       REAL,
    in_portfolio    TEXT,            -- 'single_1' / 'single_2' / 'parlay_leg_N' / null
    FOREIGN KEY (canonical_id) REFERENCES fixtures(canonical_id)
);

CREATE TABLE IF NOT EXISTS results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id    TEXT NOT NULL,
    market          TEXT NOT NULL,
    settled_at      TEXT NOT NULL,
    outcome         INTEGER NOT NULL,  -- 1 = won, 0 = lost
    closing_odds    REAL,              -- for CLV calculation
    placed_odds     REAL,
    clv             REAL,              -- closing_odds/placed_odds - 1
    beat_close      INTEGER,
    stake_kes       REAL,
    pnl_kes         REAL,
    FOREIGN KEY (canonical_id) REFERENCES fixtures(canonical_id)
);

CREATE TABLE IF NOT EXISTS league_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    league          TEXT NOT NULL,
    season          TEXT NOT NULL,    -- e.g. '2025-26'
    match_date      TEXT NOT NULL,
    home_team       TEXT,
    away_team       TEXT,
    home_goals      INTEGER,
    away_goals      INTEGER,
    total_goals     INTEGER,
    home_xg         REAL,
    away_xg         REAL,
    fetched_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS team_form (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    team            TEXT NOT NULL,
    league          TEXT NOT NULL,
    fetched_at      TEXT NOT NULL,
    expires_at      TEXT NOT NULL,    -- TTL: 24h
    league_position INTEGER,
    form_last5      TEXT,             -- e.g. 'WWLDW'
    avg_xg_for      REAL,
    avg_xga         REAL,
    days_rest       INTEGER,
    injury_count    INTEGER,
    raw_json        TEXT
);

CREATE TABLE IF NOT EXISTS ref_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    referee_name    TEXT NOT NULL,
    league          TEXT NOT NULL,
    season          TEXT NOT NULL,
    matches         INTEGER,
    red_cards_total INTEGER,
    red_card_rate   REAL,             -- per match
    fetched_at      TEXT NOT NULL
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_fixtures_date    ON fixtures(match_date);
CREATE INDEX IF NOT EXISTS idx_fixtures_league  ON fixtures(league);
CREATE INDEX IF NOT EXISTS idx_odds_canonical   ON odds(canonical_id, market);
CREATE INDEX IF NOT EXISTS idx_predictions_run  ON predictions(run_id);
CREATE INDEX IF NOT EXISTS idx_results_market   ON results(market);
CREATE INDEX IF NOT EXISTS idx_league_stats_lg  ON league_stats(league, match_date);
CREATE INDEX IF NOT EXISTS idx_team_form_team   ON team_form(team, expires_at);

CREATE TABLE IF NOT EXISTS match_stats (
    canonical_id     TEXT PRIMARY KEY,
    xg_home          REAL,
    xg_away          REAL,
    corners_home     INTEGER,
    corners_away     INTEGER,
    cards_home       INTEGER,
    cards_away       INTEGER,
    possession_home  REAL,
    possession_away  REAL,
    shots_home       INTEGER,
    shots_away       INTEGER,
    fouls_home       REAL,
    fouls_away       REAL,
    referee_id       TEXT,
    fetched_at       TEXT NOT NULL,
    source           TEXT,
    FOREIGN KEY (canonical_id) REFERENCES fixtures(canonical_id)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_date           TEXT NOT NULL,
    market                TEXT NOT NULL,
    model_version         TEXT NOT NULL DEFAULT 'v5.0',
    regime                TEXT,
    brier_score           REAL,
    log_loss              REAL,
    ece                   REAL,
    calibration_slope     REAL,
    calibration_intercept REAL,
    sharpness             REAL,
    rolling_ir            REAL,
    n_samples             INTEGER,
    edge_halflife         INTEGER,
    complexity_score      REAL,
    prune_recommended     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS signal_registry (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name TEXT NOT NULL,
    namespace   TEXT NOT NULL,
    market      TEXT NOT NULL,
    is_active   INTEGER DEFAULT 1,
    ir_contrib  REAL DEFAULT 0.0,
    corr_flag   INTEGER DEFAULT 0,
    prune_flag  INTEGER DEFAULT 0,
    added_at    TEXT NOT NULL,
    pruned_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_match_stats     ON match_stats(canonical_id);
CREATE INDEX IF NOT EXISTS idx_model_metrics   ON model_metrics(metric_date, market);
CREATE INDEX IF NOT EXISTS idx_signal_registry ON signal_registry(namespace, is_active);
"""


@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    """Create all tables if they don't exist."""
    with _conn() as con:
        con.executescript(SCHEMA)
    logger.info(f"Store initialised: {DB_PATH.resolve()}")


# ══════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════
def upsert_fixture(fixture: Dict) -> bool:
    """
    Insert or update a fixture. Returns True if new, False if already existed.
    Preserves status if already marked finished.
    """
    now = datetime.now(timezone.utc).isoformat()
    cid = fixture["canonical_id"]

    with _conn() as con:
        existing = con.execute(
            "SELECT status FROM fixtures WHERE canonical_id = ?", (cid,)
        ).fetchone()

        if existing:
            # Don't overwrite a finished fixture's status
            if existing["status"] == "finished":
                return False
            con.execute("""
                UPDATE fixtures SET kickoff_utc=?, status=?, source=?, fetched_at=?, raw_json=?
                WHERE canonical_id=?
            """, (
                fixture.get("kickoff_utc"), fixture.get("status","unplayed"),
                fixture.get("source"), now, json.dumps(fixture), cid
            ))
            return False
        else:
            con.execute("""
                INSERT INTO fixtures (canonical_id, home, away, league, kickoff_utc,
                                      match_date, status, source, fetched_at, raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                cid, fixture["home"], fixture["away"], fixture["league"],
                fixture.get("kickoff_utc",""),
                fixture.get("match_date", fixture.get("kickoff_utc","")[:10]),
                fixture.get("status","unplayed"), fixture.get("source"),
                now, json.dumps(fixture)
            ))
            return True


def get_todays_fixtures(date_str: Optional[str] = None) -> List[Dict]:
    """Return all unplayed fixtures for today (or given date)."""
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _conn() as con:
        rows = con.execute("""
            SELECT * FROM fixtures
            WHERE match_date = ? AND status = 'unplayed'
            ORDER BY kickoff_utc
        """, (date_str,)).fetchall()
    return [dict(r) for r in rows]


def stale_fixture_guard(fixtures: List[Dict], cutoff_minutes: int = 5) -> List[Dict]:
    """
    Given a list of fixture dicts (from get_todays_fixtures or ESPN fetch),
    filter out any whose kickoff is in the past (already started/finished).

    Side effect: marks past fixtures as 'started' in the DB so they won't
    appear in future store-cache hits.

    Args:
        fixtures:        list of fixture dicts, each with 'kickoff_utc' and 'canonical_id'
        cutoff_minutes:  how many minutes before kickoff to stop predicting.
                         Default 5 — drop any match kicking off within 5 minutes or already started.

    Returns:
        Filtered list containing only fixtures that haven't kicked off yet.
    """
    now     = datetime.now(timezone.utc)
    future  = []
    stale   = []

    for fx in fixtures:
        kickoff_str = fx.get("kickoff_utc", "")
        cid         = fx.get("canonical_id", "")

        if not kickoff_str:
            # No kickoff time — keep it but log a warning
            future.append(fx)
            continue

        try:
            # Parse "YYYY-MM-DD HH:MM UTC" or ISO format
            if kickoff_str.endswith(" UTC"):
                kickoff_dt = datetime.strptime(kickoff_str, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc)
            else:
                kickoff_dt = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            # Unparseable — keep it to avoid silently dropping valid fixtures
            future.append(fx)
            continue

        minutes_until = (kickoff_dt - now).total_seconds() / 60

        if minutes_until < cutoff_minutes:
            stale.append((cid, kickoff_str, minutes_until))
        else:
            future.append(fx)

    # Mark stale fixtures in DB so they're excluded from future cache hits
    if stale:
        with _conn() as con:
            for cid, ko, mins in stale:
                status = "live" if mins > -105 else "finished"  # ~90min + extra time
                con.execute(
                    "UPDATE fixtures SET status=? WHERE canonical_id=?",
                    (status, cid)
                )

    return future, stale


def mark_fixture_finished(canonical_id: str, status: str = "finished"):
    with _conn() as con:
        con.execute(
            "UPDATE fixtures SET status=? WHERE canonical_id=?",
            (status, canonical_id)
        )


def fixture_exists_today(canonical_id: str) -> bool:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _conn() as con:
        r = con.execute(
            "SELECT 1 FROM fixtures WHERE canonical_id=? AND match_date=?",
            (canonical_id, today)
        ).fetchone()
    return r is not None


# ══════════════════════════════════════════════
# Odds
# ══════════════════════════════════════════════
def save_odds_snapshot(canonical_id: str, market: str, odds_decimal: float,
                       source: str = "", is_opening: bool = False):
    """Save an odds snapshot. Automatically detects if it's the opening price."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        # Check if we already have odds today for this market
        existing = con.execute("""
            SELECT COUNT(*) as cnt FROM odds
            WHERE canonical_id=? AND market=?
            AND date(snapshot_time) = date('now')
        """, (canonical_id, market)).fetchone()

        is_first = existing["cnt"] == 0
        con.execute("""
            INSERT INTO odds (canonical_id, snapshot_time, market, odds_decimal,
                             source, is_opening)
            VALUES (?,?,?,?,?,?)
        """, (canonical_id, now, market, odds_decimal, source, 1 if is_first else 0))


def get_opening_odds(canonical_id: str, market: str) -> Optional[float]:
    with _conn() as con:
        r = con.execute("""
            SELECT odds_decimal FROM odds
            WHERE canonical_id=? AND market=? AND is_opening=1
            ORDER BY snapshot_time ASC LIMIT 1
        """, (canonical_id, market)).fetchone()
    return r["odds_decimal"] if r else None


def get_latest_odds(canonical_id: str, market: str) -> Optional[float]:
    with _conn() as con:
        r = con.execute("""
            SELECT odds_decimal FROM odds
            WHERE canonical_id=? AND market=?
            ORDER BY snapshot_time DESC LIMIT 1
        """, (canonical_id, market)).fetchone()
    return r["odds_decimal"] if r else None


def get_line_movement(canonical_id: str, market: str) -> Optional[float]:
    """Returns (current - opening) / opening as % drift. Positive = odds drifted out."""
    opening = get_opening_odds(canonical_id, market)
    current = get_latest_odds(canonical_id, market)
    if opening and current and opening > 0:
        return round((current - opening) / opening * 100, 2)
    return None


# ══════════════════════════════════════════════
# Predictions
# ══════════════════════════════════════════════
def save_prediction(pred: Dict, run_id: str, in_portfolio: Optional[str] = None):
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute("""
            INSERT INTO predictions
            (canonical_id, run_id, generated_at, market, odds, p_true, p_market,
             p_model, ev_pct, regime, drs, motivation_delta, blend_weight,
             stake_pct, in_portfolio)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pred.get("match_id",""), run_id, now,
            pred.get("market"), pred.get("odds"), pred.get("p_true"),
            pred.get("p_market"), pred.get("p_model"), pred.get("ev_pct"),
            pred.get("regime"), pred.get("drs"), pred.get("motivation_delta"),
            pred.get("blend_weight"), pred.get("stake_pct"), in_portfolio
        ))


def save_portfolio_predictions(portfolio: Dict, run_id: str):
    """Tag predictions with their portfolio role (single_1, parlay_leg_1, etc.)"""
    for i, s in enumerate(portfolio.get("singles", []), 1):
        save_prediction(s, run_id, in_portfolio=f"single_{i}")
    parlay = portfolio.get("parlay")
    if parlay:
        for i, leg in enumerate(parlay.get("legs_detail", []), 1):
            save_prediction(leg, run_id, in_portfolio=f"parlay_leg_{i}")


# ══════════════════════════════════════════════
# Results (settle bets after match)
# ══════════════════════════════════════════════
def settle_result(canonical_id: str, market: str, outcome: int,
                  closing_odds: float, placed_odds: float,
                  stake_kes: float, pnl_kes: float):
    """
    Record a settled result. Call this after the match finishes.
    outcome: 1 = won, 0 = lost
    """
    now = datetime.now(timezone.utc).isoformat()
    clv = round(placed_odds / closing_odds - 1, 4) if closing_odds > 0 else None
    beat_close = 1 if (clv and clv > 0) else 0

    with _conn() as con:
        con.execute("""
            INSERT INTO results
            (canonical_id, market, settled_at, outcome, closing_odds,
             placed_odds, clv, beat_close, stake_kes, pnl_kes)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (canonical_id, market, now, outcome, closing_odds,
              placed_odds, clv, beat_close, stake_kes, pnl_kes))

    mark_fixture_finished(canonical_id)
    logger.info(f"Result saved: {canonical_id} {market} → {'WIN' if outcome else 'LOSS'} CLV={clv}")


# ══════════════════════════════════════════════
# League Stats (for regime detection)
# ══════════════════════════════════════════════
def save_match_result_for_stats(league: str, season: str, match_date: str,
                                 home: str, away: str,
                                 home_goals: int, away_goals: int,
                                 home_xg: float = None, away_xg: float = None):
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute("""
            INSERT OR IGNORE INTO league_stats
            (league, season, match_date, home_team, away_team,
             home_goals, away_goals, total_goals, home_xg, away_xg, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (league, season, match_date, home, away,
              home_goals, away_goals, home_goals+away_goals,
              home_xg, away_xg, now))


def get_league_rolling_mean(league: str, window: int = 30) -> Optional[float]:
    """
    Rolling mean goals per game for regime detection.
    Returns None if fewer than 10 records available.
    """
    with _conn() as con:
        rows = con.execute("""
            SELECT total_goals FROM league_stats
            WHERE league = ?
            ORDER BY match_date DESC
            LIMIT ?
        """, (league, window)).fetchall()

    if len(rows) < 10:
        return None
    goals = [r["total_goals"] for r in rows]
    return round(sum(goals) / len(goals), 3)


def get_league_baseline(league: str) -> Optional[float]:
    """5-season baseline mean goals per game."""
    with _conn() as con:
        r = con.execute("""
            SELECT AVG(total_goals) as mean FROM league_stats
            WHERE league = ?
        """, (league,)).fetchone()
    return round(r["mean"], 3) if r and r["mean"] else None


def classify_regime_from_store(league: str) -> str:
    """
    CUSUM-based regime classification from all stored league goal history.

    Replaces simple mean-threshold approach. Runs CUSUM over the full stored
    goal sequence for this league — same algorithm as RegimeDetector in
    predictor.py, but reading from persistent storage so state is not lost
    between runs.

    Minimum 8 matches required for CUSUM to produce a non-neutral signal;
    falls back to league prior if insufficient data.
    """
    # Fetch all stored goal totals in chronological order
    with _conn() as con:
        rows = con.execute(
            """SELECT total_goals FROM league_stats
               WHERE league=? AND total_goals IS NOT NULL
               ORDER BY match_date ASC, id ASC""",
            (league,)
        ).fetchall()

    goal_seq = [r["total_goals"] for r in rows]

    if len(goal_seq) < 8:
        # Fall back to static prior
        try:
            from predictor import LEAGUE_PRIORS
            avg = LEAGUE_PRIORS.get(league, LEAGUE_PRIORS["_default"])["avg_goals"]
        except ImportError:
            avg = 2.60
        if avg < 2.30: return "compression"
        if avg > 2.80: return "expansion"
        return "neutral"

    # Baseline: mean and std of entire history (treated as the "true" league mean)
    mu    = sum(goal_seq) / len(goal_seq)
    var   = sum((g - mu)**2 for g in goal_seq) / len(goal_seq)
    sigma = max(var**0.5, 0.30)

    # CUSUM parameters — matching RegimeDetector constants
    CUSUM_THRESHOLD = 2.5
    N_MIN_REGIME    = 4

    cp = cn = 0.0
    cur = cand = "neutral"
    cand_n = 0

    for g in goal_seq:
        z  = (g - mu) / sigma
        cp = max(0.0, cp + z - 0.5)
        cn = max(0.0, cn - z - 0.5)

        new_cand = ("expansion"   if cp > CUSUM_THRESHOLD else
                    "compression" if cn > CUSUM_THRESHOLD else "neutral")

        if new_cand == cand:
            cand_n += 1
        else:
            cand   = new_cand
            cand_n = 1

        if cand != cur and cand_n >= N_MIN_REGIME:
            cur    = cand
            cp     = cn = 0.0

    return cur


# ══════════════════════════════════════════════
# Team Form Cache
# ══════════════════════════════════════════════
def get_cached_team_form(team: str, league: str) -> Optional[Dict]:
    """Return cached form data if not expired (TTL: 24h)."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        r = con.execute("""
            SELECT * FROM team_form
            WHERE team=? AND league=? AND expires_at > ?
            ORDER BY fetched_at DESC LIMIT 1
        """, (team, league, now)).fetchone()
    return dict(r) if r else None


def save_team_form(team: str, league: str, form_data: Dict, ttl_hours: int = 24):
    now    = datetime.now(timezone.utc)
    expiry = (now + timedelta(hours=ttl_hours)).isoformat()
    with _conn() as con:
        con.execute("""
            INSERT INTO team_form
            (team, league, fetched_at, expires_at, league_position, form_last5,
             avg_xg_for, avg_xga, days_rest, injury_count, raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            team, league, now.isoformat(), expiry,
            form_data.get("position"), form_data.get("form_last5"),
            form_data.get("avg_xg_for"), form_data.get("avg_xga"),
            form_data.get("days_rest"), form_data.get("injury_count"),
            json.dumps(form_data)
        ))


# ══════════════════════════════════════════════
# Referee Stats
# ══════════════════════════════════════════════
def save_ref_stats(referee: str, league: str, season: str,
                   matches: int, red_cards: int):
    now  = datetime.now(timezone.utc).isoformat()
    rate = round(red_cards / matches, 4) if matches > 0 else 0
    with _conn() as con:
        con.execute("""
            INSERT OR REPLACE INTO ref_stats
            (referee_name, league, season, matches, red_cards_total, red_card_rate, fetched_at)
            VALUES (?,?,?,?,?,?,?)
        """, (referee, league, season, matches, red_cards, rate, now))


def get_ref_red_card_rate(referee: str, league: str) -> float:
    """Returns red card rate per match. Default 0.12 if unknown."""
    with _conn() as con:
        r = con.execute("""
            SELECT red_card_rate FROM ref_stats
            WHERE referee_name=? AND league=?
            ORDER BY fetched_at DESC LIMIT 1
        """, (referee, league)).fetchone()
    return r["red_card_rate"] if r else 0.12


# ══════════════════════════════════════════════
# Analytics queries
# ══════════════════════════════════════════════
def get_hit_rate_by_market(min_results: int = 10) -> List[Dict]:
    """Historical hit rate per market type — for pruning weak edges."""
    with _conn() as con:
        rows = con.execute("""
            SELECT market,
                   COUNT(*) as n,
                   AVG(outcome) as hit_rate,
                   AVG(pnl_kes / NULLIF(stake_kes,0)) as roi,
                   AVG(clv) as avg_clv,
                   SUM(pnl_kes) as total_pnl
            FROM results
            GROUP BY market
            HAVING COUNT(*) >= ?
            ORDER BY roi DESC
        """, (min_results,)).fetchall()
    return [dict(r) for r in rows]


def get_clv_summary(last_n: int = 200) -> Dict:
    """CLV summary for last N settled bets."""
    with _conn() as con:
        r = con.execute("""
            SELECT
                COUNT(*) as n,
                AVG(beat_close) as clv_rate,
                AVG(clv) as mean_clv,
                AVG(outcome) as hit_rate,
                SUM(pnl_kes) as total_pnl
            FROM (
                SELECT * FROM results
                ORDER BY settled_at DESC
                LIMIT ?
            )
        """, (last_n,)).fetchone()
    return dict(r) if r else {}


def get_brier_scores(last_n: int = 100) -> Dict:
    """Compare model vs market Brier scores for blend weight calibration."""
    with _conn() as con:
        rows = con.execute("""
            SELECT r.outcome, p.p_true, p.p_market
            FROM results r
            JOIN predictions p ON r.canonical_id = p.canonical_id
                               AND r.market = p.market
            ORDER BY r.settled_at DESC
            LIMIT ?
        """, (last_n,)).fetchall()

    if not rows:
        return {"model_brier": None, "market_brier": None, "n": 0}

    model_bs  = sum((r["p_true"]   - r["outcome"])**2 for r in rows) / len(rows)
    market_bs = sum((r["p_market"] - r["outcome"])**2 for r in rows) / len(rows)
    return {
        "model_brier":  round(model_bs, 5),
        "market_brier": round(market_bs, 5),
        "n":            len(rows),
        "recommended_blend": round(market_bs / (model_bs + market_bs), 3) if (model_bs+market_bs)>0 else 0.6,
    }


def storage_summary() -> str:
    """Quick stats on what's in the store."""
    with _conn() as con:
        f = con.execute("SELECT COUNT(*) as n FROM fixtures").fetchone()["n"]
        o = con.execute("SELECT COUNT(*) as n FROM odds").fetchone()["n"]
        p = con.execute("SELECT COUNT(*) as n FROM predictions").fetchone()["n"]
        r = con.execute("SELECT COUNT(*) as n FROM results").fetchone()["n"]
        ls= con.execute("SELECT COUNT(*) as n FROM league_stats").fetchone()["n"]
        tf= con.execute("SELECT COUNT(*) as n FROM team_form").fetchone()["n"]
    return (
        f"\n  📦 STORE: {DB_PATH.name}"
        f"\n  Fixtures: {f}  |  Odds snapshots: {o}  |  Predictions: {p}"
        f"\n  Results: {r}  |  League stats: {ls}  |  Team form cache: {tf}"
    )


# ══════════════════════════════════════════════
# Match Stats (xG, corners, cards, possession)
# ══════════════════════════════════════════════
def save_match_stats(canonical_id: str, stats: Dict):
    """Save post-match stats for calibration and feature building."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute("""
            INSERT OR REPLACE INTO match_stats
            (canonical_id, xg_home, xg_away, corners_home, corners_away,
             cards_home, cards_away, possession_home, possession_away,
             shots_home, shots_away, fouls_home, fouls_away,
             referee_id, fetched_at, source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            canonical_id,
            stats.get("xg_home"),    stats.get("xg_away"),
            stats.get("corners_home"),stats.get("corners_away"),
            stats.get("cards_home"), stats.get("cards_away"),
            stats.get("possession_home"),stats.get("possession_away"),
            stats.get("shots_home"), stats.get("shots_away"),
            stats.get("fouls_home"), stats.get("fouls_away"),
            stats.get("referee_id"), now, stats.get("source","")
        ))


def get_match_stats(canonical_id: str) -> Optional[Dict]:
    with _conn() as con:
        r = con.execute("SELECT * FROM match_stats WHERE canonical_id=?",
                        (canonical_id,)).fetchone()
    return dict(r) if r else None


def get_team_rolling_stats(team: str, league: str,
                            window: int = 5) -> Dict:
    """
    Rolling stats for a team from stored match_stats.
    Used to compute xG, corners, cards averages for predictor features.
    """
    with _conn() as con:
        rows = con.execute("""
            SELECT ms.xg_home, ms.xg_away, ms.corners_home, ms.corners_away,
                   ms.cards_home, ms.cards_away,
                   f.home, f.away, f.match_date
            FROM match_stats ms
            JOIN fixtures f ON ms.canonical_id = f.canonical_id
            WHERE (f.home=? OR f.away=?) AND f.league=?
            AND f.status='finished'
            ORDER BY f.match_date DESC
            LIMIT ?
        """, (team, team, league, window)).fetchall()

    if not rows:
        return {}

    xg_vals, corner_vals, card_vals = [], [], []
    for r in rows:
        is_home = r["home"] == team
        xg_vals.append(r["xg_home"] if is_home else r["xg_away"])
        corner_vals.append(r["corners_home"] if is_home else r["corners_away"])
        card_vals.append(r["cards_home"] if is_home else r["cards_away"])

    xg_vals    = [x for x in xg_vals    if x is not None]
    corner_vals= [x for x in corner_vals if x is not None]
    card_vals  = [x for x in card_vals  if x is not None]

    return {
        "avg_xg":      round(sum(xg_vals)/len(xg_vals),3)      if xg_vals    else None,
        "avg_corners": round(sum(corner_vals)/len(corner_vals),1) if corner_vals else None,
        "avg_cards":   round(sum(card_vals)/len(card_vals),2)   if card_vals  else None,
        "n_matches":   len(rows),
    }


# ══════════════════════════════════════════════
# Model Metrics
# ══════════════════════════════════════════════
def save_model_metrics(metrics_dict: Dict, model_version: str = "v5.0"):
    """Persist calibration lab metrics to model_metrics table."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with _conn() as con:
        for market, m in metrics_dict.items():
            if hasattr(m, "__dict__"):
                m = m.__dict__
            con.execute("""
                INSERT INTO model_metrics
                (metric_date, market, model_version, brier_score, log_loss,
                 ece, calibration_slope, calibration_intercept, sharpness,
                 rolling_ir, n_samples, edge_halflife, complexity_score,
                 prune_recommended)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                now, market, model_version,
                m.get("brier_score"),  m.get("log_loss"),
                m.get("ece"),          m.get("cal_slope"),
                m.get("cal_intercept"),m.get("sharpness"),
                m.get("rolling_ir"),   m.get("n"),
                m.get("edge_halflife"),m.get("complexity_score"),
                1 if m.get("prune_recommended") else 0,
            ))


def get_latest_cal_slopes(model_version: str = "v5.0") -> Dict[str, float]:
    """Return most recent calibration slope per market."""
    with _conn() as con:
        rows = con.execute("""
            SELECT market, calibration_slope
            FROM model_metrics
            WHERE model_version=? AND calibration_slope IS NOT NULL
            GROUP BY market
            HAVING metric_date = MAX(metric_date)
        """, (model_version,)).fetchall()
    return {r["market"]: r["calibration_slope"] for r in rows}


# ══════════════════════════════════════════════
# Signal Registry
# ══════════════════════════════════════════════
def register_signal(name: str, namespace: str, market: str, ir_contrib: float = 0.0):
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute("""
            INSERT OR IGNORE INTO signal_registry
            (signal_name, namespace, market, is_active, ir_contrib, added_at)
            VALUES (?,?,?,1,?,?)
        """, (name, namespace, market, ir_contrib, now))


def prune_signal(name: str, namespace: str, reason: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as con:
        con.execute("""
            UPDATE signal_registry
            SET is_active=0, prune_flag=1, pruned_at=?
            WHERE signal_name=? AND namespace=?
        """, (now, name, namespace))
    logger.info(f"Signal pruned: [{namespace}] {name}  ({reason})")


def get_active_signals(namespace: str) -> List[str]:
    with _conn() as con:
        rows = con.execute("""
            SELECT signal_name FROM signal_registry
            WHERE namespace=? AND is_active=1
            ORDER BY ir_contrib DESC
        """, (namespace,)).fetchall()
    return [r["signal_name"] for r in rows]


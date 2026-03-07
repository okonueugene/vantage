"""
Deep diagnostic: traces the exact path the engine takes for AEK Athens,
including enrich_fixture output and raw predictor results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from store import init_db, _conn
from run_predictions import enrich_fixture
from predictor import Predictor

init_db()

# Pull the AEK fixture directly from DB
with _conn() as con:
    fx = con.execute(
        "SELECT * FROM fixtures WHERE canonical_id='AEKA_LARI_20260307'"
    ).fetchone()

if not fx:
    print("AEK fixture not in DB — searching by team name")
    with _conn() as con:
        fx = con.execute(
            "SELECT * FROM fixtures WHERE home LIKE '%AEK%' OR home LIKE '%Aek%' LIMIT 1"
        ).fetchone()

if not fx:
    print("ERROR: Cannot find AEK fixture in DB")
    sys.exit(1)

fx_dict = dict(fx)
print("=== RAW FIXTURE FROM DB ===")
for k, v in fx_dict.items():
    print(f"  {k}: {v}")

print()
print("=== AFTER enrich_fixture ===")
enriched = enrich_fixture(fx_dict)
for k, v in enriched.items():
    if v is not None:
        print(f"  {k}: {v}")

# Build the row exactly as run_predictions_pipeline does
odds_map = enriched.get("odds", {})
row = {
    "home": enriched["home"], "away": enriched["away"],
    "league": enriched["league"], "regime": enriched["regime"],
    "drs": enriched["drs"], "motivation_delta": enriched["motivation_delta"],
    "odds_1":      odds_map.get("home_win"),
    "odds_X":      odds_map.get("draw"),
    "odds_2":      odds_map.get("away_win"),
    "odds_over25": odds_map.get("over25"),
    "odds_u25":    odds_map.get("under25"),
    "home_xg":         enriched.get("home_xg"),
    "away_xg":         enriched.get("away_xg"),
    "league_rolling_goals": enriched.get("league_rolling_goals"),
}
row = {k: v for k, v in row.items() if v is not None}

print()
print("=== ROW PASSED TO PREDICTOR ===")
for k, v in row.items():
    print(f"  {k}: {v}")

print()
print("=== PREDICTOR OUTPUT ===")
p = Predictor()
for pred in p.predict(row):
    if pred.market in ("home_win", "draw", "away_win"):
        raw = pred.p_true - pred.p_market
        print(f"  {pred.market:<12} p_model={pred.p_model:.4f}  p_market={pred.p_market:.4f}  "
              f"p_true={pred.p_true:.4f}  raw={raw:.4f}  EV={pred.ev_pct:.1f}%  "
              f"blend={pred.blend_weight:.2f}  has_edge={pred.has_edge}")
        if pred.market == "draw":
            print(f"    -> draw blend expected 0.20 (no team data), got {pred.blend_weight:.2f}")
            if pred.has_edge:
                print("    !! STILL PASSING — investigating why !!")
            else:
                print("    -> Correctly filtered")

# Also check if odds are coming from the DB snapshot (may differ from what we tested with)
print()
print("=== ODDS SNAPSHOT IN DB ===")
with _conn() as con:
    snaps = con.execute(
        "SELECT market, odds, timestamp FROM odds_snapshots WHERE canonical_id='AEKA_LARI_20260307' ORDER BY timestamp DESC LIMIT 5"
    ).fetchall()
    for s in snaps:
        print(f"  {dict(s)}")

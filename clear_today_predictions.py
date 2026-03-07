"""Run once to clear today's stale predictions so the engine re-runs clean."""
from store import _conn, init_db

init_db()
with _conn() as con:
    deleted = con.execute(
        "DELETE FROM predictions WHERE generated_at > '2026-03-07'"
    ).rowcount
    print(f"Cleared {deleted} stale predictions")

    # Also confirm which predictor version is active
    try:
        from predictor import Predictor
        p = Predictor()
        row = {
            "home": "AEK Athens", "away": "Larissa FC",
            "league": "Super League Greece",
            "regime": "neutral", "drs": 0.12, "motivation_delta": 0,
            "odds_1": 1.35, "odds_X": 6.50, "odds_2": 8.00,
        }
        preds = p.predict(row)
        for pr in preds:
            if pr.market == "draw":
                print(f"Draw prediction: p_true={pr.p_true:.3f}  EV={pr.ev_pct:.1f}%  has_edge={pr.has_edge}")
                if pr.p_true < 0.20:
                    print("NEW predictor confirmed (p_true < 20%) -- draw will be filtered")
                else:
                    print("WARNING: still running OLD predictor (p_true >= 20%) -- check file copy")
    except Exception as e:
        print(f"Predictor check failed: {e}")

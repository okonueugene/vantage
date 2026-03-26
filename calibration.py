"""
calibration.py  —  Vantage Engine v5.0
───────────────────────────────────────
Forecast Validation Lab + Signal Pruning Engine

Tracks per-market:
  • Brier score
  • Log loss
  • ECE (Expected Calibration Error) — reliability curve
  • Calibration slope / intercept
  • Sharpness (variance of predicted probabilities)
  • Rolling IR  =  Mean(prediction_error) / Std(prediction_error)

Edge Decay Monitor:
  • EdgeHalfLife = matches until rolling Brier deteriorates 50% from peak
  • If half-life < HALFLIFE_MIN → prune signal

Complexity Governor:
  • ComplexityScore = ActiveSignals / RollingIR
  • If score rises without IR improvement → recommend pruning

Monte Carlo (Forecast Quality):
  • Simulates football seasons at the FORECAST level — not bankroll
  • Tracks Brier distribution, regime misclassification, error clustering
  • No stakes, no Kelly — pure model quality measurement

All results feed back into Predictor.update_calibration() to
apply per-market calibration slopes.
"""

import logging, math, json, random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("Vantage.Calibration")

HALFLIFE_MIN      = 100    # bets — warn if edge half-life below this
BRIER_WINDOW      = 50     # rolling window for Brier / IR
ECE_BINS          = 10     # reliability curve bins
COMPLEXITY_MAX    = 15.0   # prune if ActiveSignals/IR exceeds this


# ─── Dataclasses ──────────────────────────────────────────────────────────────
@dataclass
class PredRecord:
    """One settled prediction for calibration tracking."""
    market:    str
    p_true:    float     # model's predicted probability
    outcome:   int       # 1 = event occurred, 0 = did not
    model_ver: str = "v5.0"
    regime:    str = "neutral"
    settled_at: str = ""


@dataclass
class MarketMetrics:
    market:             str
    n:                  int   = 0
    brier_score:        float = 0.0
    log_loss:           float = 0.0
    ece:                float = 0.0
    cal_slope:          float = 1.0
    cal_intercept:      float = 0.0
    sharpness:          float = 0.0
    rolling_ir:         float = 0.0
    hit_rate:           float = 0.0
    edge_halflife:      Optional[int] = None
    complexity_score:   float = 0.0
    prune_recommended:  bool  = False


@dataclass
class MonteCarloForecastResult:
    """One simulated season — forecast quality metrics only."""
    n_matches:             int
    mean_brier:            float
    brier_p5:              float
    brier_p95:             float
    regime_misclass_rate:  float
    error_cluster_mag:     float   # avg size of consecutive bad predictions
    forecast_drawdown:     float   # max cumulative prediction deviation


# ─── CalibrationLab ───────────────────────────────────────────────────────────
class CalibrationLab:

    def __init__(self, log_path: str = "calibration_log.jsonl"):
        self.log_path = Path(log_path)
        self._records:  Dict[str, List[PredRecord]] = {}  # market → records
        self._active_signals: Dict[str, int] = {}         # namespace → count
        self._load()

    # ─── Record a prediction result ───────────────────────────────────────────
    def record(self, market: str, p_true: float, outcome: int,
               regime: str = "neutral", model_ver: str = "v5.0"):
        rec = PredRecord(
            market=market, p_true=p_true, outcome=outcome,
            regime=regime, model_ver=model_ver,
            settled_at=datetime.now(timezone.utc).isoformat()
        )
        self._records.setdefault(market, []).append(rec)
        self._append_log(rec)

    def set_active_signals(self, namespace: str, count: int):
        self._active_signals[namespace] = count

    # ─── Compute all metrics for one market ───────────────────────────────────
    def compute_metrics(self, market: str, window: int = BRIER_WINDOW) -> MarketMetrics:
        recs = self._records.get(market, [])
        if not recs:
            return MarketMetrics(market=market)

        recent = recs[-window:]
        n = len(recent)
        probs   = [r.p_true  for r in recent]
        outcomes= [r.outcome for r in recent]

        brier    = self._brier(probs, outcomes)
        ll       = self._log_loss(probs, outcomes)
        ece      = self._ece(probs, outcomes)
        slope, intercept = self._cal_slope(probs, outcomes)
        sharp    = self._sharpness(probs)
        ir       = self._rolling_ir(probs, outcomes)
        hit_rate = sum(outcomes)/n if n>0 else 0.0
        hl       = self._edge_halflife(market)
        ns       = self._namespace(market)
        n_sigs   = self._active_signals.get(ns, 1)
        complexity = n_sigs / max(ir, 0.01)
        # Don't prune on tiny samples — IR is meaningless below n=20
        prune    = n >= 20 and (complexity > COMPLEXITY_MAX or (hl is not None and hl < HALFLIFE_MIN))

        if prune:
            logger.warning(f"[{market}] PRUNE RECOMMENDED: "
                           f"complexity={complexity:.1f}, halflife={hl}")

        return MarketMetrics(
            market=market, n=n,
            brier_score=round(brier,5), log_loss=round(ll,5),
            ece=round(ece,5), cal_slope=round(slope,4),
            cal_intercept=round(intercept,4), sharpness=round(sharp,5),
            rolling_ir=round(ir,4), hit_rate=round(hit_rate,4),
            edge_halflife=hl, complexity_score=round(complexity,2),
            prune_recommended=prune,
        )

    def all_metrics(self) -> Dict[str, MarketMetrics]:
        return {m: self.compute_metrics(m) for m in self._records}

    # ─── Brier score ──────────────────────────────────────────────────────────
    def _brier(self, probs, outcomes) -> float:
        if not probs: return 1.0
        return sum((p-o)**2 for p,o in zip(probs,outcomes)) / len(probs)

    # ─── Log loss ─────────────────────────────────────────────────────────────
    def _log_loss(self, probs, outcomes) -> float:
        if not probs: return 10.0
        eps = 1e-7
        return -sum(o*math.log(max(p,eps))+(1-o)*math.log(max(1-p,eps))
                    for p,o in zip(probs,outcomes)) / len(probs)

    # ─── ECE (Expected Calibration Error) ─────────────────────────────────────
    def _ece(self, probs, outcomes, n_bins=ECE_BINS) -> float:
        """
        Partition predictions into n_bins by confidence.
        ECE = Σ (|bin| / N) × |accuracy(bin) - confidence(bin)|
        Low ECE = well-calibrated model.
        """
        n   = len(probs)
        if n == 0: return 1.0
        bins = [[] for _ in range(n_bins)]
        for p, o in zip(probs, outcomes):
            idx = min(int(p * n_bins), n_bins-1)
            bins[idx].append((p, o))
        ece = 0.0
        for b in bins:
            if not b: continue
            conf = sum(p for p,_ in b) / len(b)
            acc  = sum(o for _,o in b) / len(b)
            ece += (len(b)/n) * abs(acc - conf)
        return ece

    # ─── Calibration slope via isotonic regression (linear approx) ────────────
    def _cal_slope(self, probs, outcomes) -> Tuple[float, float]:
        """
        Fit linear: outcome ~ slope*p + intercept
        slope=1, intercept=0 → perfect calibration
        slope<1 → overconfident, slope>1 → underconfident
        """
        n = len(probs)
        if n < 10: return 1.0, 0.0
        mp   = sum(probs)/n
        mo   = sum(outcomes)/n
        cov  = sum((probs[i]-mp)*(outcomes[i]-mo) for i in range(n))/n
        var  = sum((p-mp)**2 for p in probs)/n
        if var < 1e-9: return 1.0, 0.0
        slope     = cov / var
        intercept = mo - slope * mp
        return round(slope, 4), round(intercept, 4)

    # ─── Sharpness ────────────────────────────────────────────────────────────
    def _sharpness(self, probs) -> float:
        """Variance of predicted probabilities. Higher = more decisive model."""
        if len(probs) < 2: return 0.0
        mean = sum(probs)/len(probs)
        return sum((p-mean)**2 for p in probs)/len(probs)

    # ─── Rolling IR (Forecast Quality) ───────────────────────────────────────
    def _rolling_ir(self, probs, outcomes) -> float:
        """
        IR = Mean(prediction_error) / Std(prediction_error)
        prediction_error = p_true - outcome  (signed)
        High IR → consistent, low-variance errors (model has systematic signal)
        Low/negative IR → noisy, unstable predictions
        """
        if len(probs) < 5: return 0.0
        errors = [p-o for p,o in zip(probs, outcomes)]
        mean_e = sum(errors)/len(errors)
        std_e  = (sum((e-mean_e)**2 for e in errors)/len(errors))**0.5
        if std_e < 1e-9: return 0.0
        return round(mean_e / std_e, 4)

    # ─── Edge half-life ───────────────────────────────────────────────────────
    def _edge_halflife(self, market: str) -> Optional[int]:
        """
        Find how many bets until rolling Brier deteriorated 50% from its peak.
        Returns None if not enough data or peak not yet degrading.
        """
        recs = self._records.get(market, [])
        if len(recs) < BRIER_WINDOW * 2: return None

        # Compute rolling Brier scores in windows
        scores = []
        for i in range(BRIER_WINDOW, len(recs)):
            window = recs[i-BRIER_WINDOW:i]
            bs = self._brier([r.p_true for r in window],
                              [r.outcome for r in window])
            scores.append(bs)

        if not scores: return None
        peak_brier = min(scores)         # lower = better
        double_brier = peak_brier * 2    # 100% deterioration = half-life proxy

        peak_idx = scores.index(peak_brier)
        for i in range(peak_idx, len(scores)):
            if scores[i] >= double_brier:
                return i - peak_idx      # bets from peak to half-life
        return None

    # ─── Namespace mapping ────────────────────────────────────────────────────
    def _namespace(self, market: str) -> str:
        if market in ("home_win","draw","away_win","over25","under25"): return "goals"
        if market.startswith("corners"): return "corners"
        if market.startswith("cards"):   return "cards"
        return "goals"

    # ─── Full report ─────────────────────────────────────────────────────────
    def report(self) -> str:
        lines = ["\n  ── CALIBRATION LAB REPORT ──────────────────────────────────"]
        all_m = self.all_metrics()
        if not all_m:
            return "  No settled predictions yet."
        for mkt, m in sorted(all_m.items()):
            lines.append(f"\n  [{mkt.upper()}]  n={m.n}")
            lines.append(f"    Brier: {m.brier_score:.4f}  LogLoss: {m.log_loss:.4f}  ECE: {m.ece:.4f}")
            lines.append(f"    CalSlope: {m.cal_slope:.3f}  Intercept: {m.cal_intercept:.4f}")
            lines.append(f"    Sharpness: {m.sharpness:.4f}  IR: {m.rolling_ir:.4f}  HitRate: {m.hit_rate:.3f}")
            if m.edge_halflife:
                lines.append(f"    EdgeHalfLife: {m.edge_halflife} bets")
            if m.prune_recommended:
                lines.append(f"    ⚠️  PRUNE RECOMMENDED (complexity={m.complexity_score:.1f})")
        lines.append("\n  ──────────────────────────────────────────────────────────")
        return "\n".join(lines)

    # ─── Persistence ──────────────────────────────────────────────────────────
    def _append_log(self, rec: PredRecord):
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec.__dict__) + "\n")
        except Exception as e:
            logger.warning(f"Calibration log write failed: {e}")

    def _load(self):
        if not self.log_path.exists(): return
        try:
            with open(self.log_path) as f:
                for line in f:
                    try:
                        d = json.loads(line.strip())
                        rec = PredRecord(**d)
                        self._records.setdefault(rec.market, []).append(rec)
                    except Exception:
                        continue
            total = sum(len(v) for v in self._records.values())
            logger.info(f"Calibration: loaded {total} records across "
                        f"{len(self._records)} markets")
        except Exception as e:
            logger.warning(f"Calibration load failed: {e}")


# ─── Monte Carlo: delegate to montecarlo.py ───────────────────────────────────
# The authoritative Monte Carlo implementation lives in montecarlo.py.
# This stub preserves any code that imports ForecastMonteCarloSim from here.
class ForecastMonteCarloSim:
    """
    Thin wrapper — delegates to MonteCarloSimulator in montecarlo.py.
    montecarlo.py is the authoritative implementation with the full
    7-step pipeline (latent params → team strengths → Poisson simulation →
    structural breaks → regime injection → predictor → forecast degradation).
    """
    def __init__(self, n_seasons: int = 10_000, **kwargs):
        from montecarlo import MonteCarloSimulator
        self._sim = MonteCarloSimulator(n_seasons=n_seasons)

    def run(self):
        return self._sim.run()


if __name__ == "__main__":
    # Run the authoritative Monte Carlo directly
    from montecarlo import MonteCarloSimulator
    import argparse
    parser = argparse.ArgumentParser(description="Calibration Lab")
    parser.add_argument("--mc", action="store_true", help="Run Monte Carlo forecast quality")
    parser.add_argument("--seasons", type=int, default=10_000)
    args = parser.parse_args()
    if args.mc:
        MonteCarloSimulator(n_seasons=args.seasons).run()
    else:
        lab = CalibrationLab()
        print(lab.report())


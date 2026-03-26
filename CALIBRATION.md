# Brier calibration — practical path

The engine uses a **blend weight** to combine model probabilities with market-implied probabilities:  
`p_true = blend × p_model + (1 − blend) × p_market`.  
That weight is calibrated from **settled predictions** so the blend minimizes combined Brier score.

## Current behaviour

- **0 settled results**  
  Blend is fixed at **0.60** (default).  
  Meta-error no longer penalizes “small sample” when there are **0** results (only when 1–19 results in that league).

- **1–29 settled results**  
  Blend stays **0.60**; Brier is computed and logged so you can see progress.

- **30+ settled results**  
  Blend is computed as:
  - `w = (Brier_market − Brier_model) / (Brier_market − Brier_model + ε)`, clamped to [0.25, 0.95].
  - Model much better → blend ≈ 0.8–0.95.  
  - Market better → blend ≈ 0.25–0.5.

- **200+ settled results**  
  Calibration is treated as **stable** (logged and in meta).

## Steps to start real calibration

### 1. Collect settled predictions + outcomes

- Every run already **saves predictions** to the store (canonical_id, market, p_true, p_market, odds, etc.).
- You need to **record outcomes** after matches: for each bet you placed, store **won** or **lost** (and optionally closing odds, stake, PnL).

### 2. Record results (three options)

**Option A — Settle one bet (lookup from last prediction)**  
Use when you have a single result and the bet was from a recent run:

```bash
python run_predictions.py settle-simple CANONICAL_ID MARKET won
# or: lost
# Optional: closing_odds (default = placed odds), stake_kes (default = 100)
python run_predictions.py settle-simple ARSE_CHEL_20260307 over25 lost 2.02 150
```

**Option B — Settle with full numbers**  
Use when you have exact closing odds, stake and PnL:

```bash
python run_predictions.py settle CANONICAL_ID MARKET 1 CLOSING_ODDS PLACED_ODDS STAKE_KES PNL_KES
# outcome: 1 = won, 0 = lost
```

**Option C — Bulk import from CSV**  
Use for backfilling or importing from a spreadsheet:

```bash
python run_predictions.py import-results results.csv
```

CSV must have a **header** and columns:  
`canonical_id`, `market`, `outcome`, `closing_odds`, `placed_odds`, `stake_kes`, `pnl_kes`  
(outcome can be 1/0 or won/lost).

### 3. Check calibration

```bash
python run_predictions.py stats
```

Shows Brier scores, blend weight, and whether you have enough data (30+ for blend to move, 200+ for stable).

### 4. Let the engine use the new blend

On the **next** run of `python run_predictions.py`, the pipeline loads `get_brier_scores()` and applies `recommended_blend` (once n ≥ 30). No extra step.

## Target sample sizes

- **30+** settled bets → blend starts updating from Brier (no longer stuck at 0.6).
- **200–500+** per market → stable, reliable blend and calibration.

## Notes

- **Brier join:** Results are joined to predictions on `(canonical_id, market)`. The latest matching prediction is used. For bulk imports, ensure those matches were previously predicted (or add predictions first).
- **Vig:** Market Brier uses the **stored** p_market from the prediction (already vig-adjusted when the bet was made).
- **Meta-error:** With 0 results we no longer add the “small sample” penalty, so stakes are less conservative when you have no history yet.

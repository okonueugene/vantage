# Markets: What’s modeled vs what appears on the slip

## Short answer

- **Draws:** The system is **not** biased toward draws. Draws are **capped** (max 1 per slip, 1.5% stake) and need **higher EV** to qualify. You see draw tips often because **only 1X2 odds are usually available** (ESPN), so the only edges we can output are home/draw/away; the market often underprices draws, so the model finds edge there.
- **Over/under, corners, BTTS:** Over/under 2.5 is **modeled** but ESPN rarely gives O/U odds. **Corners** and **cards** are **modeled** but we don’t have a live odds feed for them in the main pipeline, so they almost never appear. **BTTS** is **not implemented** (no model, no odds).

---

## What the predictor models

| Market        | Model | Odds source in main pipeline | In slip? |
|---------------|-------|------------------------------|----------|
| home_win      | Yes   | ESPN (1X2)                   | Yes      |
| draw          | Yes   | ESPN (1X2)                   | Yes (capped) |
| away_win      | Yes   | ESPN (1X2)                   | Yes      |
| over25        | Yes   | ESPN (often missing)         | When odds exist |
| under25       | Yes   | ESPN (often missing)          | When odds exist |
| corners_over  | Yes   | Not provided (optional feed)  | When odds exist |
| cards_over    | Yes   | Not provided (optional feed)  | When odds exist |
| BTTS          | No    | —                            | No       |

---

## Draw handling (not pro-draw)

- **Portfolio:** Max **1** draw bet per slip; draw stake capped at **1.5%** of bankroll.
- **Predictor:** Draw needs **higher EV** to count as edge (e.g. ≥7% for odds ≤4, ≥18% for odds >4). With no team xG, draw is blended more with the market (0.20) than home/away (0.30), so we don’t over-trust the prior.
- So the design is **restrictive** on draws, not biased toward them. Many draw tips appear because 1X2 is the only market we usually have odds for, and draws are often value vs the market.

---

## Why corners / O-U / BTTS rarely or never appear

1. **Odds source:** The main pipeline uses the **ESPN** scoreboard API, which only exposes **1X2** (and sometimes not even over/under 2.5). So:
   - **Over/under 2.5:** Model runs only when `odds_over25` / `odds_u25` exist (e.g. from cache or another feed).
   - **Corners / cards:** The predictor computes probabilities, but we only add them to the slip when **odds** are present. The pipeline now passes `odds_corners_over` and `odds_cards_over` from fixture `odds` when you provide them (e.g. from `odds_fetcher` or another feed). Until then, corners/cards edges are never selected.

2. **BTTS:** There is no BTTS model or odds mapping yet. Adding it would require a BTTS probability model and a BTTS odds source.

---

## How to get more markets on the slip

- **OddsFetcher (The Odds API / API-Football):** The main pipeline (`run_predictions.py`) now calls the odds fetcher when keys are set. Add to `.env`:
  - `THE_ODDS_API_KEY=your_key` (or `THE_ODDS_API_KEYS=key1,key2`) — free tier at the-odds-api.com
  - Optional: `API_FOOTBALL_KEY=your_key` for API-Football odds
  Then run `python run_predictions.py` as usual. Fixtures in supported leagues get 1X2 + over/under + corners/cards (when the API returns them) merged into `fixture["odds"]` and saved to the store. More leagues can be added in `odds_fetcher.py` → `THEODDS_SPORT_KEYS`.
- **Over/under 2.5:** If not using OddsFetcher, ensure fixtures have `odds["over25"]` and `odds["under25"]` from another feed. The pipeline and predictor already support them.
- **Corners / cards:** OddsFetcher can supply `corners_over` and `cards_over` when the API supports those markets. Otherwise add them to fixture `odds` from another source; the pipeline passes them through so the predictor can emit edges.
- **BTTS:** OddsFetcher may return `btts_yes`; a BTTS probability model is not yet implemented, so BTTS odds are stored but not yet used for edge detection.

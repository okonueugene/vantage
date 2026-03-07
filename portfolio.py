"""
portfolio.py
────────────
Dual-output portfolio constructor.

Output A: 2 high-conviction singles  (stability, CLV capture)
Output B: 1 controlled 4-leg parlay  (convex payoff, diversified edge)

Key rules:
  - Singles and parlay built from DIFFERENT logic (not just top-4 EV)
  - No team appears twice across portfolio
  - Max 1 leg per league, max 1 per market type in parlay
  - Singles: 3% bankroll each (half-Kelly capped)
  - Parlay:  1.5% flat (sized like an option, NOT Kelly)
  - DRS cap: parlay vetoed if >2 legs have DRS > 0.20
  - Regime cap: parlay size cut 30% if >50% legs in compression
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("Vantage.Portfolio")

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BANKROLL_DEFAULT      = 10_000   # KES — override in run_engine.py

# Singles
SINGLE_KELLY_FRACTION = 0.50     # half-Kelly
SINGLE_MAX_PCT        = 0.030    # 3% hard cap (full-signal matches)
SINGLE_MAX_PCT_LONG   = 0.015    # 1.5% cap for odds > 3.5 (less certainty)
SINGLE_MAX_PCT_VLONG  = 0.010    # 1.0% cap for odds > 5.0 (speculative)
SINGLE_MIN_EV         = 3.0
SINGLE_MIN_ODDS       = 1.40
SINGLE_MAX_ODDS       = 8.00     # No hard cut — taper stake instead
SINGLE_MAX_DRS        = 0.35

# Draw market constraints (post-match review v4.5)
DRAW_MAX_PCT          = 0.015    # 1.5% hard cap regardless of EV or Kelly
MAX_DRAWS_PER_SLIP    = 1        # max 1 draw bet across entire portfolio

# Minimum EV scales with odds — longer shots need more edge to justify
# EV_MIN(odds) = SINGLE_MIN_EV + max(0, (odds - 3.0) * 1.5)
# e.g. odds=4.0 → min EV = 4.5%,  odds=6.0 → min EV = 7.5%
SINGLE_EV_ODDS_SCALE  = 1.5

# Parlay
PARLAY_FLAT_PCT       = 0.015    # 1.5% flat
PARLAY_MIN_EV         = 2.5
PARLAY_AVG_ODDS_MIN   = 1.50     # loosened — more candidates on thin days
PARLAY_AVG_ODDS_MAX   = 3.50     # raised — allow value at longer prices
PARLAY_MAX_LEG_DRS    = 0.20
PARLAY_DRS_VETO       = 2
PARLAY_COMPRESSION_THRESHOLD = 0.50

# Dynamic leg count
PARLAY_4_LEGS_EV_THRESHOLD = 7.0
PARLAY_3_LEGS_EV_THRESHOLD = 5.0


@dataclass
class SingleBet:
    home: str
    away: str
    league: str
    market: str
    p_true: float
    decimal_odds: float
    ev_pct: float
    drs: float
    stake_units: float   # fraction of bankroll
    stake_kes: float
    regime: str
    rationale: str = ""


@dataclass
class ParlayLeg:
    home: str
    away: str
    league: str
    market: str
    p_true: float
    decimal_odds: float
    ev_pct: float
    drs: float
    regime: str


@dataclass
class ParlayBet:
    legs: List[ParlayLeg]
    combined_odds: float
    combined_p_true: float
    combined_ev_pct: float
    stake_units: float
    stake_kes: float
    drs_flag: bool = False
    compression_flag: bool = False
    size_reduction: float = 1.0   # multiplier applied to stake
    veto_reason: str = ""
    is_valid: bool = True


@dataclass
class Portfolio:
    singles: List[SingleBet]
    parlay: Optional[ParlayBet]
    bankroll: float
    total_exposure_pct: float
    total_exposure_kes: float
    regime_exposure: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class PortfolioBuilder:

    def __init__(self, bankroll: float = BANKROLL_DEFAULT):
        self.bankroll = bankroll

    # ──────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────
    def build(self, pred_df: pd.DataFrame) -> Portfolio:
        """
        Takes the edge-filtered prediction DataFrame from Predictor.predict_all()
        and constructs the dual-output portfolio.
        """
        if pred_df.empty:
            logger.warning("Portfolio builder received empty candidate pool")
            return Portfolio(singles=[], parlay=None, bankroll=self.bankroll,
                           total_exposure_pct=0, total_exposure_kes=0)

        # Filter to eligible candidates
        candidates = self._filter_candidates(pred_df)
        logger.info(f"Portfolio candidates after filter: {len(candidates)}")

        if len(candidates) < 2:
            logger.warning("Fewer than 2 valid candidates — cannot build full portfolio")

        # Build singles first (they set exclusion constraints for parlay)
        singles = self._select_singles(candidates)
        used_teams = {s.home for s in singles} | {s.away for s in singles}
        used_leagues = {s.league for s in singles}
        used_markets = {s.market for s in singles}

        # Build parlay from remaining candidates
        parlay = self._build_parlay(
            candidates,
            exclude_teams=used_teams,
            exclude_leagues=used_leagues,
            exclude_markets=used_markets,
        )

        # Portfolio totals
        single_stake = sum(s.stake_kes for s in singles)
        parlay_stake = parlay.stake_kes if parlay and parlay.is_valid else 0
        total_kes    = single_stake + parlay_stake
        total_pct    = total_kes / self.bankroll

        # Regime exposure
        all_regimes = [s.regime for s in singles]
        if parlay and parlay.is_valid:
            all_regimes += [l.regime for l in parlay.legs]
        regime_exposure = {r: all_regimes.count(r) / len(all_regimes)
                          for r in set(all_regimes)} if all_regimes else {}

        return Portfolio(
            singles=singles,
            parlay=parlay if (parlay and parlay.is_valid) else None,
            bankroll=self.bankroll,
            total_exposure_pct=round(total_pct, 4),
            total_exposure_kes=round(total_kes, 2),
            regime_exposure=regime_exposure,
        )

    # ──────────────────────────────────────────────
    # Step 1: Filter candidate pool
    # ──────────────────────────────────────────────
    def _filter_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to eligible candidates with tiered odds handling.

        Instead of a hard odds cap, we:
          - Accept any odds >= SINGLE_MIN_ODDS
          - Scale minimum required EV upward with odds (longer shots need more edge)
          - Apply tiered stake caps downstream in _make_single
          - Still remove compression overs and DRS-busted matches
        """
        df = df.copy()

        # Minimum odds floor (removes near-certainties where edge is noise)
        df = df[df["decimal_odds"] >= SINGLE_MIN_ODDS]

        # Hard DRS cap
        df = df[df["drs"] <= SINGLE_MAX_DRS]

        # Scaled EV minimum: EV_min = BASE + max(0, (odds - 3.0) * scale)
        # odds=2.0 → min 3%,  odds=4.0 → min 4.5%,  odds=6.0 → min 7.5%
        df["ev_min_required"] = SINGLE_MIN_EV + (
            (df["decimal_odds"] - 3.0).clip(lower=0) * SINGLE_EV_ODDS_SCALE
        )
        df = df[df["ev_pct"] >= df["ev_min_required"]]

        # No overs in compression
        compression_overs = (df["regime"] == "compression") & (df["market"] == "over25")
        df = df[~compression_overs]

        # Liquidity weight: top leagues = 1.0, others = 0.85
        top_leagues = {
            "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1",
            "Champions League", "Europa League", "Eredivisie",
        }
        df["liquidity_weight"] = df["league"].apply(
            lambda l: 1.0 if l in top_leagues else 0.85
        )

        # Odds tier label (used by _make_single for stake sizing)
        def _tier(o):
            if o > 5.0: return "very_long"
            if o > 3.5: return "long"
            return "normal"
        df["odds_tier"] = df["decimal_odds"].apply(_tier)

        # Composite score: EV × (1 - DRS) × liquidity × tier_weight
        tier_weight = df["odds_tier"].map({"normal": 1.0, "long": 0.80, "very_long": 0.60})
        df["selection_score"] = df["ev_pct"] * (1 - df["drs"]) * df["liquidity_weight"] * tier_weight

        df = df.sort_values("selection_score", ascending=False).reset_index(drop=True)
        return df

    # ──────────────────────────────────────────────
    # Step 2: Select 2 singles
    # ──────────────────────────────────────────────
    def _select_singles(self, df: pd.DataFrame) -> List[SingleBet]:
        """
        Select top 2 candidates: different leagues preferred, different markets preferred.
        Hard rule: max 1 draw bet per slip. On thin days relax league diversity.
        """
        selected   = []
        used_leagues = set()
        draws_used = 0

        # Pass 1: full diversity + draw cap
        for _, row in df.iterrows():
            if len(selected) >= 2: break
            if row["league"] in used_leagues: continue
            if row["market"] == "draw" and draws_used >= MAX_DRAWS_PER_SLIP: continue
            bet = self._make_single(row)
            selected.append(bet)
            used_leagues.add(row["league"])
            if row["market"] == "draw":
                draws_used += 1

        # Pass 2: relax league constraint, keep draw cap
        if len(selected) < 2:
            for _, row in df.iterrows():
                if len(selected) >= 2: break
                if row["market"] == "draw" and draws_used >= MAX_DRAWS_PER_SLIP: continue
                already = any(
                    s.home == row["home"] and s.away == row["away"]
                    and s.market == row["market"] for s in selected
                )
                if not already:
                    bet = self._make_single(row)
                    selected.append(bet)
                    if row["market"] == "draw":
                        draws_used += 1

        logger.info(f"Singles selected: {len(selected)}"
                    + (f" (draws: {draws_used})" if draws_used else ""))
        return selected

    def _make_single(self, row) -> SingleBet:
        """Half-Kelly stake with tiered cap. Draws capped at DRAW_MAX_PCT regardless."""
        p    = float(row["p_true"])
        odds = float(row["decimal_odds"])
        tier = row.get("odds_tier", "normal")

        full_kelly = (p * odds - 1) / (odds - 1) if (odds - 1) > 0 else 0
        half_kelly = full_kelly * SINGLE_KELLY_FRACTION

        # Tiered cap: longer odds = less bankroll committed
        if tier == "very_long":
            max_pct = SINGLE_MAX_PCT_VLONG   # 1.0%
        elif tier == "long":
            max_pct = SINGLE_MAX_PCT_LONG    # 1.5%
        else:
            max_pct = SINGLE_MAX_PCT         # 3.0%

        # Draw override: hard 1.5% cap regardless of tier or Kelly
        if row.get("market") == "draw":
            max_pct = min(max_pct, DRAW_MAX_PCT)

        stake_pct = min(max(half_kelly, 0.005), max_pct)
        stake_kes = round(stake_pct * self.bankroll, 2)

        rationale = (
            f"EV {row['ev_pct']:.1f}% | odds {odds:.2f} ({tier}) | "
            f"DRS {row['drs']:.2f} | "
            f"p_true {row['p_true']:.3f} vs p_market {row['p_market']:.3f} | "
            f"Regime: {row['regime']}"
            + (" | DRAW cap 1.5%" if row.get("market") == "draw" else "")
        )

        return SingleBet(
            home=row["home"], away=row["away"],
            league=row["league"], market=row["market"],
            p_true=float(row["p_true"]),
            decimal_odds=float(row["decimal_odds"]),
            ev_pct=float(row["ev_pct"]),
            drs=float(row["drs"]),
            stake_units=round(stake_pct, 4),
            stake_kes=stake_kes,
            regime=row["regime"],
            rationale=rationale,
        )

    def _build_parlay(
        self,
        df: pd.DataFrame,
        exclude_teams: set,
        exclude_leagues: set,
        exclude_markets: set,
    ) -> Optional[ParlayBet]:
        """
        Build parlay from candidates NOT used in singles.
        Rules:
          - No overlap with singles teams
          - Max 1 per league (CAN reuse market types here — different matches)
          - Max 1 per market type
          - Average odds 1.65–2.20
          - Dynamic leg count based on avg EV
        """
        # Exclude singles matches
        available = df[
            ~(df["home"].isin(exclude_teams) | df["away"].isin(exclude_teams))
        ].copy()

        if available.empty:
            logger.info("No candidates available for parlay (all used in singles)")
            return self._empty_parlay("No eligible candidates after singles exclusion")

        used_p_leagues = set()
        used_p_markets = set()
        legs = []

        for _, row in available.iterrows():
            if len(legs) >= 4:
                break

            league = row["league"]
            market = row["market"]

            # Structural diversity: 1 per league, 1 per market
            if league in used_p_leagues:
                continue
            if market in used_p_markets:
                continue

            # Parlay-specific odds range (wider than before — value exists at longer prices)
            if not (PARLAY_AVG_ODDS_MIN <= float(row["decimal_odds"]) <= PARLAY_AVG_ODDS_MAX):
                continue

            legs.append(ParlayLeg(
                home=row["home"], away=row["away"],
                league=row["league"], market=row["market"],
                p_true=float(row["p_true"]),
                decimal_odds=float(row["decimal_odds"]),
                ev_pct=float(row["ev_pct"]),
                drs=float(row["drs"]),
                regime=row["regime"],
            ))
            used_p_leagues.add(league)
            used_p_markets.add(market)

        if len(legs) < 2:
            return self._empty_parlay(f"Only {len(legs)} valid parlay legs found (need ≥2)")

        # Dynamic leg count
        avg_ev = sum(l.ev_pct for l in legs) / len(legs)
        if avg_ev >= PARLAY_4_LEGS_EV_THRESHOLD:
            target_legs = 4
        elif avg_ev >= PARLAY_3_LEGS_EV_THRESHOLD:
            target_legs = 3
        else:
            target_legs = 2
        legs = legs[:target_legs]
        logger.info(f"Parlay: {len(legs)} legs (avg EV {avg_ev:.1f}% → target {target_legs})")

        # Veto checks
        high_drs_count = sum(1 for l in legs if l.drs > PARLAY_MAX_LEG_DRS)
        drs_veto = high_drs_count >= PARLAY_DRS_VETO
        if drs_veto:
            return self._empty_parlay(
                f"Parlay vetoed: {high_drs_count} legs have DRS > {PARLAY_MAX_LEG_DRS}"
            )

        compression_count = sum(1 for l in legs if l.regime == "compression")
        compression_flag  = compression_count / len(legs) > PARLAY_COMPRESSION_THRESHOLD
        size_reduction    = 0.70 if compression_flag else 1.0

        # Combined stats
        combined_odds    = round(
            1.0 * __import__('math').prod(l.decimal_odds for l in legs), 3
        )
        combined_p_true  = round(
            1.0 * __import__('math').prod(l.p_true for l in legs), 6
        )
        combined_ev_pct  = round((combined_p_true * combined_odds - 1) * 100, 2)

        # Flat stake (treat like an option)
        base_stake = PARLAY_FLAT_PCT * self.bankroll * size_reduction
        stake_kes  = round(base_stake, 2)

        return ParlayBet(
            legs=legs,
            combined_odds=combined_odds,
            combined_p_true=combined_p_true,
            combined_ev_pct=combined_ev_pct,
            stake_units=round(PARLAY_FLAT_PCT * size_reduction, 4),
            stake_kes=stake_kes,
            drs_flag=high_drs_count > 0,
            compression_flag=compression_flag,
            size_reduction=size_reduction,
            is_valid=True,
        )

    def _empty_parlay(self, reason: str) -> ParlayBet:
        logger.info(f"Parlay not built: {reason}")
        return ParlayBet(
            legs=[], combined_odds=0, combined_p_true=0,
            combined_ev_pct=0, stake_units=0, stake_kes=0,
            is_valid=False, veto_reason=reason,
        )

    # ──────────────────────────────────────────────
    # Shared exposure check
    # ──────────────────────────────────────────────
    def check_shared_exposure(self, portfolio: Portfolio) -> List[str]:
        """Warn if any team appears more than once across singles + parlay."""
        all_teams = []
        for s in portfolio.singles:
            all_teams += [s.home, s.away]
        if portfolio.parlay:
            for l in portfolio.parlay.legs:
                all_teams += [l.home, l.away]

        seen = set()
        duplicates = []
        for t in all_teams:
            if t in seen:
                duplicates.append(t)
            seen.add(t)
        return duplicates

"""
predictor.py  —  Vantage Engine v5.0
─────────────────────────────────────
Multi-market probabilistic models with isolated signal namespaces.

Market models
─────────────
  Goals / 1X2  →  Hierarchical Poisson (λ~Gamma prior) + Dixon-Coles correction
  Over/Under   →  Derived from Poisson joint — coherent with 1X2
  Corners      →  Negative Binomial  (overdispersed count data)
  Cards        →  Zero-Inflated Poisson  (many low-card matches)

GameTempo (latent)
──────────────────
  Shared latent variable. Prevents incoherent outputs:
    e.g.  high goals p  +  ultra-low corners p  →  flagged

Regime (CUSUM)
──────────────
  Bayesian change-point detection. Switches only when:
    (a) CUSUM exceeds threshold  AND
    (b) New regime has persisted ≥ N_MIN_REGIME matches
  Bounded |δ_regime| ≤ 0.10

Signal orthogonality
────────────────────
  Corr(Si, Sj) < 0.30 required within namespace
  Weaker signal rejected if violation detected
"""

import logging, math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger("Vantage.Predictor")

EV_THRESHOLD  = 3.0
N_MIN_REGIME  = 4       # matches before regime switch confirmed
CORR_THRESH   = 0.30    # max allowed pairwise signal correlation

# ─── League priors ────────────────────────────────────────────────────────────
# alpha/beta → Gamma prior on team λ  (avg_goals = alpha/beta)
# dc_rho     → Dixon-Coles correlation correction (negative = under-count 0-0)
LEAGUE_PRIORS = {
    # ── England ───────────────────────────────────────────────────────────────
    "Premier League":      {"alpha":5.30,"beta":2.00,"avg_goals":2.65,"home_adv":0.08,
                            "dc_rho":-0.10,"corners_mu":10.8,"corners_k":8.0,
                            "cards_lambda":3.2,"cards_pi":0.05},
    # Championship: physical, high-tempo, strong home crowd effect
    "Championship":        {"alpha":5.50,"beta":2.00,"avg_goals":2.75,"home_adv":0.11,
                            "dc_rho":-0.09,"corners_mu":11.2,"corners_k":8.2,
                            "cards_lambda":3.8,"cards_pi":0.04},
    # League One: more direct, higher scoring than Championship
    "League One":          {"alpha":5.60,"beta":2.00,"avg_goals":2.80,"home_adv":0.12,
                            "dc_rho":-0.08,"corners_mu":11.0,"corners_k":8.0,
                            "cards_lambda":3.9,"cards_pi":0.04},
    # League Two: direct play, strong home advantage, physical
    "League Two":          {"alpha":5.40,"beta":2.00,"avg_goals":2.70,"home_adv":0.13,
                            "dc_rho":-0.09,"corners_mu":10.5,"corners_k":7.5,
                            "cards_lambda":4.0,"cards_pi":0.04},
    # ── Spain ─────────────────────────────────────────────────────────────────
    "La Liga":             {"alpha":5.16,"beta":2.00,"avg_goals":2.58,"home_adv":0.09,
                            "dc_rho":-0.09,"corners_mu":10.2,"corners_k":7.5,
                            "cards_lambda":3.8,"cards_pi":0.04},
    # Segunda: slightly more goals than La Liga, stronger home effect
    "Segunda Division":    {"alpha":5.20,"beta":2.00,"avg_goals":2.60,"home_adv":0.11,
                            "dc_rho":-0.10,"corners_mu":10.0,"corners_k":7.2,
                            "cards_lambda":4.0,"cards_pi":0.04},
    # ── Germany ───────────────────────────────────────────────────────────────
    "Bundesliga":          {"alpha":5.90,"beta":2.00,"avg_goals":2.95,"home_adv":0.07,
                            "dc_rho":-0.08,"corners_mu":11.5,"corners_k":9.0,
                            "cards_lambda":2.9,"cards_pi":0.06},
    # 2. Bundesliga: similar tempo, slightly higher goals, stronger home adv
    "2. Bundesliga":       {"alpha":5.80,"beta":2.00,"avg_goals":2.90,"home_adv":0.09,
                            "dc_rho":-0.08,"corners_mu":11.2,"corners_k":8.5,
                            "cards_lambda":3.1,"cards_pi":0.06},
    # ── Italy ─────────────────────────────────────────────────────────────────
    "Serie A":             {"alpha":5.04,"beta":2.00,"avg_goals":2.52,"home_adv":0.10,
                            "dc_rho":-0.12,"corners_mu":10.4,"corners_k":7.8,
                            "cards_lambda":4.1,"cards_pi":0.04},
    # Serie B: more direct, slightly more goals, very card-heavy
    "Serie B":             {"alpha":5.10,"beta":2.00,"avg_goals":2.55,"home_adv":0.11,
                            "dc_rho":-0.11,"corners_mu":10.2,"corners_k":7.5,
                            "cards_lambda":4.5,"cards_pi":0.04},
    # ── France ────────────────────────────────────────────────────────────────
    "Ligue 1":             {"alpha":4.96,"beta":2.00,"avg_goals":2.48,"home_adv":0.08,
                            "dc_rho":-0.10,"corners_mu":9.8, "corners_k":7.2,
                            "cards_lambda":3.5,"cards_pi":0.05},
    # Ligue 2: more physical, more goals, stronger home advantage
    "Ligue 2":             {"alpha":5.30,"beta":2.00,"avg_goals":2.65,"home_adv":0.10,
                            "dc_rho":-0.09,"corners_mu":10.2,"corners_k":7.5,
                            "cards_lambda":3.8,"cards_pi":0.05},
    # ── Netherlands ───────────────────────────────────────────────────────────
    "Eredivisie":          {"alpha":6.10,"beta":2.00,"avg_goals":3.05,"home_adv":0.07,
                            "dc_rho":-0.07,"corners_mu":12.0,"corners_k":9.5,
                            "cards_lambda":2.8,"cards_pi":0.07},
    # ── Portugal ──────────────────────────────────────────────────────────────
    "Primeira Liga":       {"alpha":4.90,"beta":2.00,"avg_goals":2.45,"home_adv":0.11,
                            "dc_rho":-0.11,"corners_mu":9.5, "corners_k":7.0,
                            "cards_lambda":3.9,"cards_pi":0.04},
    # ── Turkey ────────────────────────────────────────────────────────────────
    "Super Lig":           {"alpha":5.40,"beta":2.00,"avg_goals":2.70,"home_adv":0.12,
                            "dc_rho":-0.10,"corners_mu":10.5,"corners_k":7.5,
                            "cards_lambda":4.2,"cards_pi":0.03},
    "Süper Lig":           {"alpha":5.40,"beta":2.00,"avg_goals":2.70,"home_adv":0.12,
                            "dc_rho":-0.10,"corners_mu":10.5,"corners_k":7.5,
                            "cards_lambda":4.2,"cards_pi":0.03},
    # ── Scotland ──────────────────────────────────────────────────────────────
    "Scottish Premiership":{"alpha":4.80,"beta":2.00,"avg_goals":2.40,"home_adv":0.13,
                            "dc_rho":-0.12,"corners_mu":10.0,"corners_k":7.0,
                            "cards_lambda":3.6,"cards_pi":0.05},
    # Scottish Championship: more physical, higher scoring than Premiership
    "Scottish Championship":{"alpha":5.00,"beta":2.00,"avg_goals":2.50,"home_adv":0.14,
                             "dc_rho":-0.11,"corners_mu":10.2,"corners_k":7.2,
                             "cards_lambda":3.8,"cards_pi":0.05},
    # ── Belgium ───────────────────────────────────────────────────────────────
    "Belgian Pro League":  {"alpha":5.70,"beta":2.00,"avg_goals":2.85,"home_adv":0.09,
                            "dc_rho":-0.08,"corners_mu":11.2,"corners_k":8.5,
                            "cards_lambda":3.1,"cards_pi":0.06},
    # ── Greece ────────────────────────────────────────────────────────────────
    # Physical play, strong home advantage, card-heavy
    "Super League Greece":  {"alpha":4.90,"beta":2.00,"avg_goals":2.45,"home_adv":0.13,
                             "dc_rho":-0.11,"corners_mu":9.8,"corners_k":7.0,
                             "cards_lambda":4.3,"cards_pi":0.04},
    # ── Scandinavia ───────────────────────────────────────────────────────────
    # Danish Superliga: balanced, moderate scoring, good home advantage
    "Danish Superliga":    {"alpha":5.30,"beta":2.00,"avg_goals":2.65,"home_adv":0.10,
                            "dc_rho":-0.09,"corners_mu":10.5,"corners_k":7.8,
                            "cards_lambda":3.0,"cards_pi":0.06},
    # Allsvenskan: open attacking play, strong summer home crowds
    "Allsvenskan":         {"alpha":5.20,"beta":2.00,"avg_goals":2.60,"home_adv":0.10,
                            "dc_rho":-0.09,"corners_mu":10.8,"corners_k":8.0,
                            "cards_lambda":3.2,"cards_pi":0.06},
    # Eliteserien: high-tempo, decent scoring, physical
    "Eliteserien":         {"alpha":5.10,"beta":2.00,"avg_goals":2.55,"home_adv":0.11,
                            "dc_rho":-0.10,"corners_mu":10.5,"corners_k":7.8,
                            "cards_lambda":3.3,"cards_pi":0.05},
    # ── Central/Western Europe ────────────────────────────────────────────────
    # Swiss Super League: tactically tight, lower scoring
    "Swiss Super League":  {"alpha":4.80,"beta":2.00,"avg_goals":2.40,"home_adv":0.09,
                            "dc_rho":-0.11,"corners_mu":9.8,"corners_k":7.2,
                            "cards_lambda":3.2,"cards_pi":0.05},
    # Austrian Bundesliga: similar to Swiss but slightly more open
    "Austrian Bundesliga": {"alpha":5.30,"beta":2.00,"avg_goals":2.65,"home_adv":0.10,
                            "dc_rho":-0.09,"corners_mu":10.5,"corners_k":7.8,
                            "cards_lambda":3.4,"cards_pi":0.05},
    # Czech First League: physical, moderate scoring, strong home factor
    "Czech First League":  {"alpha":5.00,"beta":2.00,"avg_goals":2.50,"home_adv":0.12,
                            "dc_rho":-0.11,"corners_mu":10.0,"corners_k":7.2,
                            "cards_lambda":3.8,"cards_pi":0.05},
    # Ekstraklasa: competitive, physical, decent goals
    "Ekstraklasa":         {"alpha":5.10,"beta":2.00,"avg_goals":2.55,"home_adv":0.12,
                            "dc_rho":-0.10,"corners_mu":10.2,"corners_k":7.5,
                            "cards_lambda":3.9,"cards_pi":0.05},
    # ── Eastern Europe ────────────────────────────────────────────────────────
    # Ukrainian Premier League: pre-war data; variable quality, strong home adv
    "Ukrainian Premier League": {"alpha":4.80,"beta":2.00,"avg_goals":2.40,"home_adv":0.12,
                                 "dc_rho":-0.12,"corners_mu":9.5,"corners_k":7.0,
                                 "cards_lambda":4.0,"cards_pi":0.05},
    # Russian Premier League: low scoring, defensive, very strong home advantage
    "Russian Premier League":   {"alpha":4.60,"beta":2.00,"avg_goals":2.30,"home_adv":0.14,
                                  "dc_rho":-0.13,"corners_mu":9.2,"corners_k":6.8,
                                  "cards_lambda":3.8,"cards_pi":0.05},
    # ── Americas ──────────────────────────────────────────────────────────────
    "MLS":                 {"alpha":5.60,"beta":2.00,"avg_goals":2.80,"home_adv":0.06,
                            "dc_rho":-0.08,"corners_mu":10.8,"corners_k":8.0,
                            "cards_lambda":3.4,"cards_pi":0.05},
    # Liga MX: high scoring, very strong home advantage, partisan crowds
    "Liga MX":             {"alpha":5.50,"beta":2.00,"avg_goals":2.75,"home_adv":0.13,
                            "dc_rho":-0.09,"corners_mu":10.5,"corners_k":7.8,
                            "cards_lambda":4.0,"cards_pi":0.04},
    # Brasileirao: high scoring, strong home advantage, physical
    "Brasileirao":         {"alpha":5.40,"beta":2.00,"avg_goals":2.70,"home_adv":0.12,
                            "dc_rho":-0.09,"corners_mu":10.8,"corners_k":8.0,
                            "cards_lambda":4.2,"cards_pi":0.04},
    # Argentine Primera: defensive quality, lower scoring than Brazil
    "Argentine Primera":   {"alpha":5.00,"beta":2.00,"avg_goals":2.50,"home_adv":0.13,
                            "dc_rho":-0.11,"corners_mu":9.8,"corners_k":7.2,
                            "cards_lambda":4.3,"cards_pi":0.04},
    # ── Asia / Middle East / Oceania ──────────────────────────────────────────
    "Saudi Pro League":    {"alpha":5.50,"beta":2.00,"avg_goals":2.75,"home_adv":0.14,
                            "dc_rho":-0.09,"corners_mu":10.6,"corners_k":7.8,
                            "cards_lambda":4.0,"cards_pi":0.04},
    "A-League":            {"alpha":5.40,"beta":2.00,"avg_goals":2.70,"home_adv":0.08,
                            "dc_rho":-0.09,"corners_mu":10.4,"corners_k":7.6,
                            "cards_lambda":3.3,"cards_pi":0.05},
    # J1 League: technical, lower scoring, disciplined, weak home advantage
    "J1 League":           {"alpha":4.70,"beta":2.00,"avg_goals":2.35,"home_adv":0.07,
                            "dc_rho":-0.12,"corners_mu":9.5,"corners_k":7.2,
                            "cards_lambda":2.6,"cards_pi":0.07},
    # Chinese Super League: variable quality, strong home advantage
    "Chinese Super League":{"alpha":5.20,"beta":2.00,"avg_goals":2.60,"home_adv":0.11,
                            "dc_rho":-0.10,"corners_mu":10.2,"corners_k":7.5,
                            "cards_lambda":3.5,"cards_pi":0.05},
    # ── UEFA competitions ─────────────────────────────────────────────────────
    "Champions League":    {"alpha":5.60,"beta":2.00,"avg_goals":2.80,"home_adv":0.05,
                            "dc_rho":-0.08,"corners_mu":11.0,"corners_k":8.5,
                            "cards_lambda":3.0,"cards_pi":0.06},
    "Europa League":       {"alpha":5.40,"beta":2.00,"avg_goals":2.70,"home_adv":0.04,
                            "dc_rho":-0.09,"corners_mu":10.5,"corners_k":8.0,
                            "cards_lambda":3.2,"cards_pi":0.05},
    # Conference League: lower quality clubs, open play, more goals
    "Conference League":   {"alpha":5.50,"beta":2.00,"avg_goals":2.75,"home_adv":0.04,
                            "dc_rho":-0.08,"corners_mu":10.8,"corners_k":8.0,
                            "cards_lambda":3.1,"cards_pi":0.05},
    # ── Domestic cups ─────────────────────────────────────────────────────────
    # Cup games: slightly more goals than league avg (pressure, open play, AET risk),
    # reduced home advantage (neutral venues in later rounds), more cards (high stakes).
    # Draw probability lower in knockout rounds (extra time resolves most).
    # These are used as priors when no team-specific data available.
    "FA Cup":              {"alpha":5.20,"beta":1.90,"avg_goals":2.80,"home_adv":0.07,
                            "dc_rho":-0.06,"corners_mu":10.6,"corners_k":7.5,
                            "cards_lambda":3.5,"cards_pi":0.06},
    "EFL Cup":             {"alpha":5.10,"beta":1.85,"avg_goals":2.85,"home_adv":0.07,
                            "dc_rho":-0.06,"corners_mu":10.4,"corners_k":7.5,
                            "cards_lambda":3.4,"cards_pi":0.06},
    "Copa del Rey":        {"alpha":5.30,"beta":1.95,"avg_goals":2.75,"home_adv":0.07,
                            "dc_rho":-0.07,"corners_mu":10.2,"corners_k":7.8,
                            "cards_lambda":3.6,"cards_pi":0.07},
    "DFB-Pokal":           {"alpha":5.40,"beta":2.00,"avg_goals":2.90,"home_adv":0.08,
                            "dc_rho":-0.07,"corners_mu":10.8,"corners_k":7.8,
                            "cards_lambda":3.3,"cards_pi":0.05},
    "Coppa Italia":        {"alpha":5.20,"beta":1.95,"avg_goals":2.70,"home_adv":0.07,
                            "dc_rho":-0.07,"corners_mu":10.0,"corners_k":7.5,
                            "cards_lambda":3.8,"cards_pi":0.07},
    "Coupe de France":     {"alpha":5.10,"beta":1.90,"avg_goals":2.75,"home_adv":0.08,
                            "dc_rho":-0.07,"corners_mu":10.2,"corners_k":7.5,
                            "cards_lambda":3.5,"cards_pi":0.06},
    "KNVB Cup":            {"alpha":5.30,"beta":1.95,"avg_goals":2.90,"home_adv":0.08,
                            "dc_rho":-0.07,"corners_mu":10.5,"corners_k":7.8,
                            "cards_lambda":3.2,"cards_pi":0.05},
    "Taça de Portugal":    {"alpha":5.00,"beta":1.85,"avg_goals":2.70,"home_adv":0.09,
                            "dc_rho":-0.08,"corners_mu":10.0,"corners_k":7.5,
                            "cards_lambda":3.5,"cards_pi":0.06},
    "Scottish FA Cup":     {"alpha":4.90,"beta":1.80,"avg_goals":2.65,"home_adv":0.08,
                            "dc_rho":-0.08,"corners_mu":10.0,"corners_k":7.5,
                            "cards_lambda":3.4,"cards_pi":0.06},
    "Scottish League Cup": {"alpha":4.90,"beta":1.80,"avg_goals":2.65,"home_adv":0.08,
                            "dc_rho":-0.08,"corners_mu":10.0,"corners_k":7.5,
                            "cards_lambda":3.4,"cards_pi":0.06},
    "Turkish Cup":         {"alpha":5.00,"beta":1.85,"avg_goals":2.75,"home_adv":0.08,
                            "dc_rho":-0.07,"corners_mu":10.3,"corners_k":7.8,
                            "cards_lambda":3.9,"cards_pi":0.08},
    "Belgian Cup":         {"alpha":5.20,"beta":1.90,"avg_goals":2.85,"home_adv":0.08,
                            "dc_rho":-0.07,"corners_mu":10.5,"corners_k":7.8,
                            "cards_lambda":3.3,"cards_pi":0.05},
    "Greek Cup":           {"alpha":4.90,"beta":1.80,"avg_goals":2.60,"home_adv":0.09,
                            "dc_rho":-0.08,"corners_mu":9.8,"corners_k":7.5,
                            "cards_lambda":4.0,"cards_pi":0.08},
    "Copa do Brasil":      {"alpha":5.10,"beta":1.90,"avg_goals":2.70,"home_adv":0.10,
                            "dc_rho":-0.07,"corners_mu":10.0,"corners_k":7.5,
                            "cards_lambda":3.8,"cards_pi":0.07},
    "Copa Argentina":      {"alpha":5.00,"beta":1.85,"avg_goals":2.65,"home_adv":0.10,
                            "dc_rho":-0.08,"corners_mu":9.8,"corners_k":7.5,
                            "cards_lambda":3.9,"cards_pi":0.07},
    "US Open Cup":         {"alpha":5.20,"beta":1.90,"avg_goals":2.80,"home_adv":0.08,
                            "dc_rho":-0.07,"corners_mu":10.2,"corners_k":7.5,
                            "cards_lambda":3.3,"cards_pi":0.05},
    "_default":            {"alpha":5.20,"beta":2.00,"avg_goals":2.60,"home_adv":0.08,
                            "dc_rho":-0.10,"corners_mu":10.5,"corners_k":8.0,
                            "cards_lambda":3.3,"cards_pi":0.05},
}

# Regime: P_adjusted = P_base × (1 + δ),  |δ| ≤ 0.10
REGIME_DELTA = {
    "compression": {"over25":-0.10,"under25":+0.08,"home_win":-0.04,"away_win":+0.04,
                    "draw":+0.03,"corners_over":-0.08,"cards_over":+0.05},
    "neutral":     {k:0.0 for k in ["over25","under25","home_win","away_win",
                                      "draw","corners_over","cards_over"]},
    "expansion":   {"over25":+0.10,"under25":-0.08,"home_win":+0.04,"away_win":-0.04,
                    "draw":-0.03,"corners_over":+0.08,"cards_over":-0.05},
}


# ─── Dataclasses ──────────────────────────────────────────────────────────────
@dataclass
class GameTempo:
    """Latent variable shared across markets. Ensures coherent multi-market outputs."""
    value:  float = 1.0      # 0.4=very slow, 1.0=baseline, 1.8=very fast
    source: str   = "prior"  # prior | xg_derived | possession_derived

    def goals_mult(self)   -> float: return 0.85 + 0.15 * self.value
    def corners_mult(self) -> float: return 0.80 + 0.20 * self.value
    def cards_mult(self)   -> float: return 1.20 - 0.20 * self.value


@dataclass
class MarketPrediction:
    market:        str
    p_model:       float
    p_market:      float
    p_true:        float
    p_low:         float
    p_high:        float
    ev_pct:        float
    decimal_odds:  float
    has_edge:      bool
    regime:        str
    regime_delta:  float
    game_tempo:    float
    blend_weight:  float
    cal_slope:     float = 1.0
    notes:         str   = ""


# ─── Math helpers ─────────────────────────────────────────────────────────────
def _pmf(k: int, lam: float) -> float:
    return (lam**k * math.exp(-lam)) / math.factorial(k) if lam > 0 else 0.0

def _poisson_over(lam: float, t: float) -> float:
    return round(max(1 - sum(_pmf(i, lam) for i in range(int(t)+1)), 0.0), 5)

def _dc_tau(h: int, a: int, rho: float, lh: float, la: float) -> float:
    """Dixon-Coles low-score correction."""
    if   h==0 and a==0: return 1 - lh*la*rho
    elif h==0 and a==1: return 1 + lh*rho
    elif h==1 and a==0: return 1 + la*rho
    elif h==1 and a==1: return 1 - rho
    return 1.0

def _match_probs(lh: float, la: float, rho: float, max_g: int=8) -> Dict:
    """Full Poisson + DC joint distribution → 1X2 + over/under markets."""
    hw=dr=aw=o25=o35=0.0
    for h in range(max_g+1):
        for a in range(max_g+1):
            p = max(_pmf(h,lh)*_pmf(a,la)*_dc_tau(h,a,rho,lh,la), 0.0)
            if h>a: hw+=p
            elif h<a: aw+=p
            else: dr+=p
            if h+a>2: o25+=p
            if h+a>3: o35+=p
    tot = hw+dr+aw
    if tot>0: hw/=tot; dr/=tot; aw/=tot
    return {"home_win":round(hw,5),"draw":round(dr,5),"away_win":round(aw,5),
            "over25":round(o25,5),"under25":round(1-o25,5),
            "over35":round(o35,5),"under35":round(1-o35,5)}

def _nb_over(mu: float, k: float, t: float) -> float:
    """P(X>t) for Negative Binomial(mu,k). Corners are overdispersed → NB."""
    if mu<=0: return 0.5
    p_nb=k/(k+mu); r=k; cdf=0.0
    for i in range(int(t)+1):
        lp = (math.lgamma(i+r)-math.lgamma(i+1)-math.lgamma(r)
              + i*math.log(1-p_nb)+r*math.log(p_nb))
        cdf += math.exp(lp)
    return round(max(1-cdf, 0.0), 5)

def _zip_over(lam: float, pi: float, t: float) -> float:
    """P(X>t) for Zero-Inflated Poisson(lam,pi). Cards have excess zeros → ZIP."""
    if lam<=0: return 0.0
    cdf = pi+(1-pi)*math.exp(-lam)
    for i in range(1, int(t)+1):
        cdf += (1-pi)*_pmf(i, lam)
    return round(max(1-cdf, 0.0), 5)

def _shin(odds: float, margin: float=1.05) -> float:
    """Shin vig-removal. More accurate than 1/odds."""
    if odds<=1.0: return 0.97
    return min((1/odds)/margin, 0.97)


# ─── CUSUM Regime Detector ────────────────────────────────────────────────────
class RegimeDetector:
    """
    CUSUM change-point detection for league goal environment.
    Switches regime only when CUSUM > threshold AND new regime persists ≥ N_MIN_REGIME.
    Prevents chattering.
    """
    CUSUM_THRESHOLD = 2.5

    def __init__(self, baseline_mean: float, baseline_std: float, league: str=""):
        self.mu      = baseline_mean
        self.sigma   = max(baseline_std, 0.3)
        self.league  = league
        self._cp     = 0.0   # positive CUSUM
        self._cn     = 0.0   # negative CUSUM
        self._cur    = "neutral"
        self._cand   = "neutral"
        self._cand_n = 0

    def update(self, total_goals: float) -> str:
        z = (total_goals - self.mu) / self.sigma
        self._cp = max(0, self._cp + z - 0.5)
        self._cn = max(0, self._cn - z - 0.5)
        cand = ("expansion" if self._cp > self.CUSUM_THRESHOLD else
                "compression" if self._cn > self.CUSUM_THRESHOLD else "neutral")
        if cand == self._cand:
            self._cand_n += 1
        else:
            self._cand   = cand
            self._cand_n = 1
        if self._cand != self._cur and self._cand_n >= N_MIN_REGIME:
            logger.info(f"Regime [{self.league}]: {self._cur}→{self._cand} "
                        f"(CUSUM+={self._cp:.2f} -={self._cn:.2f})")
            self._cur = self._cand
            self._cp  = self._cn = 0.0
        return self._cur

    def classify(self, recent_goals: List[float]) -> str:
        """Stateless classification from a list of recent goal totals."""
        if len(recent_goals) < 5: return "neutral"
        dev = (sum(recent_goals)/len(recent_goals) - self.mu) / self.sigma
        return "compression" if dev<-0.5 else "expansion" if dev>0.5 else "neutral"

    @property
    def regime(self) -> str: return self._cur


# ─── Signal orthogonality checker ─────────────────────────────────────────────
class OrthogonalityChecker:
    """
    Enforces Corr(Si, Sj) < 0.30 within each signal namespace.
    Greedy: accepts highest-magnitude signals first.
    With insufficient history (< 20 samples) → passes all through.
    """
    def __init__(self, threshold: float = CORR_THRESH):
        self.t   = threshold
        self._h: Dict[str, List[float]] = {}

    def record(self, name: str, val: float):
        self._h.setdefault(name, []).append(val)

    def corr(self, a: str, b: str) -> float:
        ha, hb = self._h.get(a,[]), self._h.get(b,[])
        n = min(len(ha), len(hb))
        if n < 20: return 0.0
        a_ = ha[-n:]; b_ = hb[-n:]
        ma, mb = sum(a_)/n, sum(b_)/n
        cov  = sum((a_[i]-ma)*(b_[i]-mb) for i in range(n))/n
        sa   = (sum((x-ma)**2 for x in a_)/n)**0.5
        sb   = (sum((x-mb)**2 for x in b_)/n)**0.5
        return round(cov/(sa*sb), 4) if sa and sb else 0.0

    def filter(self, signals: Dict[str, float]) -> Tuple[Dict, List[str]]:
        accepted, rejected = {}, []
        for name, val in sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True):
            if any(abs(self.corr(name, acc)) >= self.t for acc in accepted):
                rejected.append(name)
            else:
                accepted[name] = val
                self.record(name, val)
        return accepted, rejected


# ─── Main Predictor ───────────────────────────────────────────────────────────
class Predictor:
    INIT_BLEND = 0.60
    MOTIV_SCALE = 0.005

    def __init__(self):
        self.blend_weight = self.INIT_BLEND
        self._ortho_goals   = OrthogonalityChecker()
        self._ortho_corners = OrthogonalityChecker()
        self._ortho_cards   = OrthogonalityChecker()
        self._cal_slopes: Dict[str, float] = {}  # from CalibrationLab

    def predict(self, row: Dict) -> List[MarketPrediction]:
        prior  = LEAGUE_PRIORS.get(row.get("league","_default"), LEAGUE_PRIORS["_default"])
        regime = row.get("regime", "neutral")
        drs    = float(row.get("drs", 0.12))
        motiv  = int(row.get("motivation_delta", 0))

        # 1. GameTempo latent variable
        tempo = self._tempo(row, prior)

        # 2. Goals signals (with orthogonality check)
        goal_signals = {}
        if row.get("home_xg"): goal_signals["rolling_xg"] = float(row["home_xg"])
        if row.get("home_rest_days"): goal_signals["rest_days"] = float(row["home_rest_days"])/7
        if row.get("shot_conversion_delta"): goal_signals["shot_conv"] = float(row["shot_conversion_delta"])
        accepted_goals, _ = self._ortho_goals.filter(goal_signals)

        # Prior-only detection: no xG → model uninformative for 1X2
        has_team_data = bool(row.get("home_xg") and row.get("away_xg"))
        prior_only_markets = {"home_win", "draw", "away_win"}

        # Fatigue context
        home_rest = float(row.get("home_rest_days") or 4)
        away_rest = float(row.get("away_rest_days") or 4)
        home_fatigued = home_rest < 3

        # Draw-specific constants
        DRAW_MIN_EV_BOOST    = 4.0   # draws need EV >= 7% base
        DRAW_MIN_EV_LONG     = 15.0  # draws at odds > 4.0 need EV >= 18% (3 + 15)
        DRAW_FATIGUE_BOOST   = 0.035 # +3.5pp when home team fatigued
        DRAW_MOTIV_SUPPRESS  = 0.015 # -1.5pp per unit away motivation delta
        DRAW_DEF_AWAY_PENALTY = 0.06 # -6pp when away team is a known low-block side

        # Defensive away style: teams whose away identity is low-block / counter-attack.
        # These sides produce fewer draws vs big home teams — they grind narrow wins/losses.
        # Grouped by primary style. Expandable as you accumulate results.
        DEFENSIVE_AWAY_TEAMS = {
            # La Liga
            "Getafe", "Almeria", "Cadiz", "Girona",
            # Premier League
            "Burnley", "Luton", "Sheffield United", "Brentford", "Crystal Palace",
            # Serie A
            "Udinese", "Empoli", "Salernitana", "Frosinone",
            # Bundesliga
            "Heidenheim", "Darmstadt", "Mainz",
            # Ligue 1
            "Brest", "Lorient", "Metz",
            # Championship / lower
            "Millwall", "Stoke City", "Sheffield Wednesday", "Preston",
            # Atletico-style (suppress draws even vs mid-table)
            "Atletico Madrid", "Atletico de Madrid",
        }

        # 3. Team lambdas
        lh, la = self._lambdas(row, prior, tempo, accepted_goals)

        # 4. All market probabilities
        base_probs = _match_probs(lh, la, prior["dc_rho"])
        base_probs.update(self._corners(row, prior, tempo))
        base_probs.update(self._cards(row, prior, drs))

        # 5. Build per-market predictions
        results = []
        odds_map = {
            "home_win":"odds_1","draw":"odds_X","away_win":"odds_2",
            "over25":"odds_over25","under25":"odds_u25",
            "corners_over":"odds_corners_over","cards_over":"odds_cards_over",
        }
        deltas = REGIME_DELTA.get(regime, REGIME_DELTA["neutral"])

        for market, col in odds_map.items():
            odds_val = self._odds(row, col)
            if odds_val is None: continue
            p_mod = base_probs.get(market, 0.0)
            if p_mod <= 0: continue

            p_mkt = _shin(odds_val)
            delta = deltas.get(market, 0.0)

            if market in prior_only_markets and not has_team_data:
                # No xG/form — heavily market-anchored.
                # Draws get extra suppression: prior draw rate ~29% inflates EV vs market ~16%.
                # Without team data we cannot confirm the edge, so penalise further.
                effective_blend = 0.20 if market == "draw" else 0.30
            elif market in ("corners_over","cards_over","corners_under","cards_under"):
                effective_blend = 0.70
            else:
                effective_blend = self.blend_weight

            blended = effective_blend * p_mod + (1 - effective_blend) * p_mkt
            p_reg   = min(max(blended * (1 + delta), 0.02), 0.97)
            p_true  = p_reg + motiv * self.MOTIV_SCALE

            # ── Draw-specific adjustments ──────────────────────────────────
            draw_note = ""
            if market == "draw":
                # A) Fatigue boost — fatigued home team less likely to dominate
                if home_fatigued:
                    p_true += DRAW_FATIGUE_BOOST
                    draw_note += f"home_fatigue(+{DRAW_FATIGUE_BOOST:.1%}) "

                # B) Motivation suppression — away team chasing points, won't settle
                if motiv > 3:
                    suppress = min(motiv * DRAW_MOTIV_SUPPRESS, 0.06)
                    p_true  -= suppress
                    draw_note += f"away_motivated(-{suppress:.1%}) "

                # C) Defensive away style penalty — low-block sides rarely draw vs
                #    big home teams; they either nick a win or absorb a defeat.
                away_team = row.get("away", "")
                if away_team in DEFENSIVE_AWAY_TEAMS:
                    p_true -= DRAW_DEF_AWAY_PENALTY
                    draw_note += f"def_away_style(-{DRAW_DEF_AWAY_PENALTY:.1%}) "

                # D) Low-event league suppression — when league rolling goals < 2.3
                #    the game is likely to be low-xG both ways, which empirically
                #    favours narrow home/away wins over 0-0 or 1-1 draws.
                league_mu = row.get("league_rolling_goals")  # injected by enrich_fixture
                if league_mu is not None and league_mu < 2.3:
                    p_true -= 0.03
                    draw_note += f"low_event_league(-3.0%) "

            p_true = min(max(p_true, 0.02), 0.97)
            p_true = min(max(p_true * self._cal_slopes.get(market, 1.0), 0.02), 0.97)
            band   = self._band(p_mod, p_mkt, drs, market)
            ev     = round((p_true * odds_val - 1) * 100, 2)

            # ── Edge thresholds ────────────────────────────────────────────
            raw_edge = p_true - p_mkt

            # Long-odds raw-edge floor (prevents phantom prior-blend edges)
            # Prior-only blend (no xG/form) is unreliable above 4.0 — floor is deliberately tight
            if odds_val > 6.0:
                min_raw_edge = 0.06 if has_team_data else 0.12
            elif odds_val > 4.0:
                min_raw_edge = 0.04 if has_team_data else 0.09
            else:
                min_raw_edge = 0.0

            # Draw EV threshold scales with odds:
            #   odds ≤ 4.0 → need EV ≥ 7%  (base 3 + 4)
            #   odds > 4.0 → need EV ≥ 18% (base 3 + 15) — high-variance territory
            if market == "draw":
                ev_boost = DRAW_MIN_EV_LONG if odds_val > 4.0 else DRAW_MIN_EV_BOOST
            else:
                ev_boost = 0.0

            ev_threshold = EV_THRESHOLD + ev_boost
            has_edge = ev >= ev_threshold and raw_edge >= min_raw_edge

            results.append(MarketPrediction(
                market=market, p_model=round(p_mod,5), p_market=round(p_mkt,5),
                p_true=round(p_true,5),
                p_low=round(max(p_true-band,0.01),4),
                p_high=round(min(p_true+band,0.99),4),
                ev_pct=ev, decimal_odds=odds_val,
                has_edge=has_edge,
                regime=regime, regime_delta=round(delta,4),
                game_tempo=round(tempo.value,3),
                blend_weight=round(effective_blend,3),
                cal_slope=self._cal_slopes.get(market,1.0),
                notes=draw_note.strip(),
            ))

        self._coherence_check(results, row)
        return results

    def _tempo(self, row, prior) -> GameTempo:
        hx, ax = row.get("home_xg"), row.get("away_xg")
        if hx and ax:
            avg = prior["avg_goals"]
            v   = (float(hx)+float(ax))/avg if avg>0 else 1.0
            return GameTempo(round(min(max(v,0.4),1.8),3), "xg_derived")
        ph, pa = row.get("home_possession_avg"), row.get("away_possession_avg")
        if ph and pa:
            bal = 1-abs(float(ph)-float(pa))/100
            return GameTempo(round(0.8+0.4*bal,3), "possession_derived")
        return GameTempo(1.0, "prior")

    def _lambdas(self, row, prior, tempo, accepted_goals) -> Tuple[float,float]:
        avg, ha = prior["avg_goals"], prior["home_adv"]
        tm = tempo.goals_mult()
        hx, ax = row.get("home_xg"), row.get("away_xg")
        if hx and ax:
            n = float(row.get("home_n_matches", 5))
            shrink = max(1-n/20, 0.10)   # Bayesian shrinkage toward league mean
            lh = (shrink*avg/2+(1-shrink)*float(hx))*(1+ha)*tm
            la = (shrink*avg/2+(1-shrink)*float(ax))*(1-ha*0.5)*tm
        else:
            lh = (avg/2)*(1+ha)*tm
            la = (avg/2)*(1-ha*0.5)*tm
        # Apply rest-days signal if accepted
        if "rest_days" in accepted_goals:
            rd = accepted_goals["rest_days"]
            lh *= (0.95+0.05*min(rd,1.0))  # well-rested → slightly higher λ
        return round(max(lh,0.1),4), round(max(la,0.1),4)

    def _corners(self, row, prior, tempo) -> Dict:
        mu = prior["corners_mu"]; k = prior["corners_k"]
        hc, ac = row.get("home_corners_avg"), row.get("away_corners_avg")
        if hc and ac:
            mu = (float(hc)+float(ac))*tempo.corners_mult()
        else:
            mu *= tempo.corners_mult()
        signals = {}
        if row.get("home_possession_avg"):
            signals["possession_pct"] = float(row["home_possession_avg"])/50.0
        accepted, _ = self._ortho_corners.filter(signals)
        for sig, val in accepted.items():
            if sig=="possession_pct": mu *= (0.95+0.05*val)
        p = _nb_over(mu, k, 9.5)
        return {"corners_over":p, "corners_under":round(1-p,5)}

    def _cards(self, row, prior, drs) -> Dict:
        lam = prior["cards_lambda"]; pi = prior["cards_pi"]
        signals = {
            "ref_strictness": drs*25/lam,
            "derby_flag":     float(row.get("derby_flag",0)),
            "foul_rate":      (float(row.get("home_foul_rate",0))+float(row.get("away_foul_rate",0)))/2,
        }
        signals = {k:v for k,v in signals.items() if v!=0}
        accepted, _ = self._ortho_cards.filter(signals)
        for sig, val in accepted.items():
            if sig=="ref_strictness": lam *= (0.8+0.4*val)
            if sig=="derby_flag":     lam *= (1.0+0.2*val)
            if sig=="foul_rate":      lam *= (0.9+0.2*val)
        lam = max(lam, 0.5)
        p = _zip_over(lam, pi, 3.5)
        return {"cards_over":p, "cards_under":round(1-p,5)}

    def _band(self, pm, pmkt, drs, market) -> float:
        base = {"cards_over":0.11,"cards_under":0.11,
                "corners_over":0.09,"corners_under":0.09,
                "draw":0.09}.get(market, 0.07)
        return round(min(base+drs*0.12+abs(pm-pmkt)*0.18, 0.22), 4)

    def _coherence_check(self, preds, row):
        pm = {p.market:p for p in preds}
        g, co = pm.get("over25"), pm.get("corners_over")
        if g and co:
            if g.p_true>0.68 and co.p_true<0.32:
                logger.warning(f"[{row.get('home')} v {row.get('away')}] "
                               f"Coherence flag: high goals({g.p_true:.2f}) + "
                               f"low corners({co.p_true:.2f})")
            if g.p_true<0.32 and co.p_true>0.68:
                logger.warning(f"[{row.get('home')} v {row.get('away')}] "
                               f"Coherence flag: low goals({g.p_true:.2f}) + "
                               f"high corners({co.p_true:.2f})")

    def update_calibration(self, market: str, slope: float):
        self._cal_slopes[market] = max(round(slope,4), 0.5)
        logger.info(f"Cal slope updated [{market}]: {slope:.4f}")

    def _odds(self, row, col) -> Optional[float]:
        v = row.get(col)
        if v is None: return None
        try:
            f = float(v); return f if f>1.01 else None
        except: return None

    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        all_p = []
        for _, row in df.iterrows():
            for p in self.predict(row.to_dict()):
                all_p.append(p.__dict__)
        if not all_p: return pd.DataFrame()
        df2 = pd.DataFrame(all_p)
        df2 = df2[df2["has_edge"]].sort_values("ev_pct",ascending=False)
        logger.info(f"Predictor: {len(all_p)} preds → {len(df2)} with edge")
        return df2.reset_index(drop=True)

"""
normalizer.py
─────────────
Standardizes team names and data formats across all sources.

The core problem: ESPN calls them "Manchester City", TheSportsDB calls them
"Man City", API-Football calls them "Manchester City FC". Cross-source
deduplication fails unless all names resolve to a canonical form.

Solution: canonical name registry with fuzzy fallback.
"""

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger("VantageV5.Normalizer")

# ──────────────────────────────────────────────
# Canonical Name Registry
# Key   = any variant seen in the wild
# Value = canonical form used throughout the engine
# ──────────────────────────────────────────────
CANONICAL_NAMES: Dict[str, str] = {
    # Premier League
    "manchester city fc":          "Manchester City",
    "manchester city":             "Manchester City",
    "man city":                    "Manchester City",
    "manchester united fc":        "Manchester United",
    "manchester united":           "Manchester United",
    "man utd":                     "Manchester United",
    "man united":                  "Manchester United",
    "tottenham hotspur fc":        "Tottenham Hotspur",
    "tottenham hotspur":           "Tottenham Hotspur",
    "spurs":                       "Tottenham Hotspur",
    "tottenham":                   "Tottenham Hotspur",
    "newcastle united fc":         "Newcastle United",
    "newcastle united":            "Newcastle United",
    "newcastle":                   "Newcastle United",
    "brighton & hove albion fc":   "Brighton",
    "brighton & hove albion":      "Brighton",
    "brighton and hove albion":    "Brighton",
    "west ham united fc":          "West Ham United",
    "west ham united":             "West Ham United",
    "west ham":                    "West Ham United",
    "wolverhampton wanderers fc":  "Wolverhampton",
    "wolverhampton wanderers":     "Wolverhampton",
    "wolves":                      "Wolverhampton",
    "leicester city fc":           "Leicester City",
    "leicester city":              "Leicester City",
    "leicester":                   "Leicester City",
    "nottingham forest fc":        "Nottingham Forest",
    "nottingham forest":           "Nottingham Forest",
    "nott'm forest":               "Nottingham Forest",
    "sheffield united fc":         "Sheffield United",
    "sheffield united":            "Sheffield United",
    "sheffield utd":               "Sheffield United",
    "luton town fc":               "Luton Town",
    "luton town":                  "Luton Town",
    # La Liga
    "fc barcelona":                "Barcelona",
    "barcelona":                   "Barcelona",
    "real madrid cf":              "Real Madrid",
    "real madrid":                 "Real Madrid",
    "atletico madrid":             "Atletico Madrid",
    "atlético madrid":             "Atletico Madrid",
    "athletic bilbao":             "Athletic Bilbao",
    "athletic club":               "Athletic Bilbao",
    "real sociedad":               "Real Sociedad",
    "real betis":                  "Real Betis",
    "villarreal cf":               "Villarreal",
    "villarreal":                  "Villarreal",
    "rcd mallorca":                "Mallorca",
    "mallorca":                    "Mallorca",
    "girona fc":                   "Girona",
    "girona":                      "Girona",
    # Bundesliga
    "fc bayern münchen":           "Bayern Munich",
    "fc bayern munich":            "Bayern Munich",
    "bayern munich":               "Bayern Munich",
    "bayern münchen":              "Bayern Munich",
    "borussia dortmund":           "Dortmund",
    "bvb":                         "Dortmund",
    "rb leipzig":                  "RB Leipzig",
    "bayer 04 leverkusen":         "Bayer Leverkusen",
    "bayer leverkusen":            "Bayer Leverkusen",
    "eintracht frankfurt":         "Eintracht Frankfurt",
    "vfb stuttgart":               "Stuttgart",
    "stuttgart":                   "Stuttgart",
    "sc freiburg":                 "Freiburg",
    "freiburg":                    "Freiburg",
    "tsg hoffenheim":              "Hoffenheim",
    "hoffenheim":                  "Hoffenheim",
    # Serie A
    "fc internazionale milano":    "Inter Milan",
    "inter milan":                 "Inter Milan",
    "inter":                       "Inter Milan",
    "internazionale":              "Inter Milan",
    "ac milan":                    "AC Milan",
    "milan":                       "AC Milan",
    "juventus fc":                 "Juventus",
    "juventus":                    "Juventus",
    "as roma":                     "Roma",
    "roma":                        "Roma",
    "ssc napoli":                  "Napoli",
    "napoli":                      "Napoli",
    "atalanta bc":                 "Atalanta",
    "atalanta":                    "Atalanta",
    "ss lazio":                    "Lazio",
    "lazio":                       "Lazio",
    "acf fiorentina":              "Fiorentina",
    "fiorentina":                  "Fiorentina",
    "torino fc":                   "Torino",
    "torino":                      "Torino",
    "bologna fc":                  "Bologna",
    "bologna":                     "Bologna",
    # Ligue 1
    "paris saint-germain":         "PSG",
    "paris saint germain":         "PSG",
    "psg":                         "PSG",
    "olympique de marseille":      "Marseille",
    "marseille":                   "Marseille",
    "olympique lyonnais":          "Lyon",
    "lyon":                        "Lyon",
    "as monaco":                   "Monaco",
    "monaco":                      "Monaco",
    "losc lille":                  "Lille",
    "lille":                       "Lille",
    "rc lens":                     "Lens",
    "lens":                        "Lens",
    "stade rennais fc":            "Rennes",
    "rennes":                      "Rennes",
    # UCL common names
    "real madrid c.f.":            "Real Madrid",
    "fc porto":                    "Porto",
    "sl benfica":                  "Benfica",
    "ajax":                        "Ajax",
    "afc ajax":                    "Ajax",
}


def normalize_team(raw_name: str) -> str:
    """
    Resolve a raw team name to its canonical form.
    Steps:
      1. Exact match in registry (lowercased)
      2. Fuzzy match: strip FC/AFC/SC suffixes, retry
      3. Fallback: title-case the raw name
    """
    if not raw_name:
        return raw_name

    cleaned = raw_name.strip()
    key     = cleaned.lower()

    # 1. Direct lookup
    if key in CANONICAL_NAMES:
        return CANONICAL_NAMES[key]

    # 2. Strip common suffixes and retry
    stripped = re.sub(
        r'\b(fc|afc|sc|cf|ac|as|ss|rc|rb|tsg|bvb|vfb|rcd|ssc|slb|sl|f\.c\.|a\.f\.c\.)\b\.?',
        '', key, flags=re.I
    ).strip()
    if stripped in CANONICAL_NAMES:
        return CANONICAL_NAMES[stripped]

    # 3. Partial match: if registry key is contained in input or vice versa
    for reg_key, canonical in CANONICAL_NAMES.items():
        if reg_key in key or key in reg_key:
            if len(min(reg_key, key, key=len)) > 5:   # avoid spurious short matches
                return canonical

    # 4. Fallback: return cleaned title-case
    return cleaned.title()


def normalize_kickoff(iso_string: str) -> Optional[str]:
    """Normalize any ISO 8601 kickoff string to 'YYYY-MM-DD HH:MM UTC'."""
    if not iso_string:
        return None
    try:
        iso_string = iso_string.replace("Z", "+00:00")
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso_string)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return iso_string   # return as-is if unparseable


def normalize_dataframe(df) -> "pd.DataFrame":
    """
    Apply all normalization to a fixtures DataFrame:
      - Canonical team names
      - Standardised kickoff format
      - Consistent column names
      - Duplicate canonical match_id generation
    """
    import pandas as pd

    if df.empty:
        return df

    df = df.copy()

    # Normalize team names
    df["home"] = df["home"].apply(normalize_team)
    df["away"] = df["away"].apply(normalize_team)

    # Normalize kickoff
    if "kickoff_utc" in df.columns:
        df["kickoff_utc"] = df["kickoff_utc"].apply(normalize_kickoff)

    # Generate canonical match_id (independent of source)
    df["canonical_id"] = df.apply(
        lambda r: f"{re.sub(r'[^A-Za-z]', '', r['home'])[:4].upper()}"
                  f"_{re.sub(r'[^A-Za-z]', '', r['away'])[:4].upper()}"
                  f"_{str(r.get('kickoff_utc', ''))[:10].replace('-', '')}",
        axis=1
    )

    logger.debug(f"Normalization complete — {len(df)} records, {df['canonical_id'].nunique()} unique matches")
    return df

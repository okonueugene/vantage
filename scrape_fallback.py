"""
scrape_fallback.py
──────────────────
Layer 2: Light scraping — ONLY used when structured APIs return < threshold.

Rules:
  - Never scrapes first
  - Only Soccerway and LiveScore (most bot-tolerant)
  - Applies same junk filter as validator
  - Rate limits: 0.8–1.6s (not 3–6s — we're validating, not discovering)
"""

import re
import time
import random
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

import requests
import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

logger = logging.getLogger("VantageV5.Fallback")

JUNK_STRINGS = {
    "terms of use", "privacy policy", "favorites", "matches",
    "soccer scores", "choose a league", "home", "away", "score",
    "espn+", "espn deportes", "football", "soccer", "hockey",
    "sign in", "log in", "register", "settings", "language",
}


def _is_valid_team(name: str) -> bool:
    if not name or len(name) < 3 or len(name) > 45:
        return False
    if name.lower().strip() in JUNK_STRINGS:
        return False
    if re.search(r'https?://|www\.|\+\d*$|sport\s*\d+$', name, re.I):
        return False
    if len(re.findall(r'[A-Za-zÀ-ÿ]', name)) < 2:
        return False
    return True


class ScrapeFallback:

    SOURCES = [
        {"url": "https://int.soccerway.com/matches/", "name": "Soccerway",
         "container_class": re.compile(r'\bmatch\b', re.I)},
        {"url": "https://www.livescore.com/en/football/", "name": "LiveScore",
         "container_class": re.compile(r'(match|event|fixture)', re.I)},
    ]

    # Only these leagues pass through the fallback — prevents noise from
    # South African regional leagues, Argentine thirds, etc.
    TARGET_LEAGUE_PATTERNS = re.compile(
        r'Premier|La Liga|Bundesliga|Serie A|Ligue 1|Eredivisie|'
        r'Primeira Liga|Champions|Europa|Conference|Süper Lig|Super Lig',
        re.I
    )

    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.5",
        })

    def _get(self, url: str) -> Optional[str]:
        for attempt in range(2):
            try:
                time.sleep(random.uniform(0.8, 1.6))
                self.session.headers.update({"User-Agent": self.ua.random})
                resp = self.session.get(url, timeout=12)
                if resp.status_code == 200:
                    return resp.text
                logger.warning(f"[{resp.status_code}] {url} (attempt {attempt+1})")
            except Exception as e:
                logger.debug(f"Fallback fetch attempt {attempt+1} failed: {e}")
        logger.error(f"Fallback failed for {url}")
        return None

    def _parse_matches(self, html: str, source: str) -> List[Dict]:
        soup = BeautifulSoup(html, "html.parser")
        records = []
        scan_time = datetime.now(timezone.utc).isoformat()

        # Find league headers + their match containers
        league_headers = soup.find_all(
            ["h2", "h3", "h4", "span", "div"],
            string=self.TARGET_LEAGUE_PATTERNS
        )

        for header in league_headers:
            league_name = header.get_text(strip=True)
            # Find sibling/child match containers after this header
            parent = header.find_parent(["div", "section", "table"])
            if not parent:
                continue

            containers = parent.find_all(
                ["tr", "div", "li"],
                class_=re.compile(r'(match|fixture|event|row)', re.I)
            )

            for item in containers:
                try:
                    time_match = item.find(string=re.compile(r'\b\d{1,2}:\d{2}\b'))
                    time_str = time_match.strip() if time_match else "N/A"

                    # Look for team-specific elements first
                    team_els = item.find_all(
                        ["td", "span", "div", "a"],
                        class_=re.compile(r'(team|club|name|home|away)', re.I)
                    )

                    if len(team_els) >= 2:
                        home_raw = team_els[0].get_text(strip=True)
                        away_raw = team_els[1].get_text(strip=True)
                    else:
                        texts = [t.strip() for t in item.stripped_strings]
                        alpha = [t for t in texts if re.match(r'^[A-Za-zÀ-ÿ\s\.\-&\']{3,}$', t)]
                        if len(alpha) < 2:
                            continue
                        home_raw, away_raw = alpha[0], alpha[1]

                    if not _is_valid_team(home_raw) or not _is_valid_team(away_raw):
                        continue
                    if home_raw.lower() == away_raw.lower():
                        continue

                    records.append({
                        "home": home_raw,
                        "away": away_raw,
                        "league": league_name,
                        "kickoff_utc": time_str,
                        "status": "scheduled",
                        "source": source,
                        "fetch_time": scan_time,
                    })
                except Exception:
                    continue

        logger.info(f"Fallback {source}: {len(records)} valid matches found")
        return records

    def fetch(self, min_structured_count: int = 20,
              structured_df: "pd.DataFrame" = None) -> pd.DataFrame:
        """
        Only runs if structured sources returned fewer than min_structured_count matches.

        Args:
            min_structured_count: threshold below which fallback activates
            structured_df:        existing DataFrame to check count against
        """
        if structured_df is not None and len(structured_df) >= min_structured_count:
            logger.info(
                f"Fallback skipped — structured sources gave {len(structured_df)} matches "
                f"(threshold: {min_structured_count})"
            )
            return pd.DataFrame()

        logger.info(f"Fallback activated — structured count below {min_structured_count}")

        all_records = []
        for src in self.SOURCES:
            html = self._get(src["url"])
            if html:
                records = self._parse_matches(html, src["name"])
                all_records.extend(records)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = df[df["home"].apply(_is_valid_team) & df["away"].apply(_is_valid_team)]
        df = df.drop_duplicates(subset=["home", "away"])
        df = df.reset_index(drop=True)

        logger.info(f"Fallback complete: {len(df)} clean matches from {len(self.SOURCES)} sources")
        return df

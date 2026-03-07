from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import urllib.parse
import json
from typing import List, Dict, Optional

BASE = "https://fbref.com"

def get_team_slug(team_name: str) -> Optional[str]:
    """Find FBref squad slug via search page"""
    search = urllib.parse.quote(team_name)
    url = f"{BASE}/search/search.fcgi?search={search}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=45000)

        try:
            page.wait_for_selector(".search-item", timeout=15000)
            soup = BeautifulSoup(page.content(), "lxml")
            for item in soup.select("#clubs .search-item"):
                if "Female" in item.get_text():
                    continue
                link = item.select_one("a[href*='/en/squads/']")
                if link:
                    return link["href"].split("/")[3]
        except:
            if "/en/squads/" in page.url:
                return page.url.split("/")[5]

        browser.close()
    return None


def parse_shooting_table(team_slug: str, season: str = "2025-2026") -> Dict:
    """Parse match logs + footer totals"""
    url = f"{BASE}/en/squads/{team_slug}/{season}/matchlogs/all_comps/shooting/"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_selector("#matchlogs_for", timeout=30000)

        soup = BeautifulSoup(page.content(), "lxml")
        browser.close()

    # ── Footer totals (most accurate) ────────────────────────────────
    totals = {}
    footer = soup.select_one("#matchlogs_for tfoot tr")
    if footer:
        for stat in ["goals", "goals_against", "shots", "shots_on_target"]:
            cell = footer.select_one(f'[data-stat="{stat}"]')
            if cell and cell.text.strip().isdigit():
                totals[stat] = int(cell.text.strip())

    # ── Match rows ───────────────────────────────────────────────────
    matches: List[Dict] = []
    for row in soup.select("#matchlogs_for tbody tr"):
        if "thead" in row.get("class", []) or "spacer" in row.get("class", []):
            continue

        def get(stat: str) -> str:
            cell = row.select_one(f'[data-stat="{stat}"]')
            return cell.get_text(strip=True) if cell else ""

        result = get("result")
        if not result or result == "--":
            continue  # skip future/unplayed

        matches.append({
            "date": get("date"),
            "comp": get("comp"),
            "venue": get("venue"),
            "result": result,
            "gf": int(get("goals") or 0),
            "ga": int(get("goals_against") or 0),
            "shots": int(get("shots") or 0),
            "sot": int(get("shots_on_target") or 0),
        })

    # ── Compute summary ──────────────────────────────────────────────
    games = len(matches)
    if games == 0:
        return {"error": "No completed matches found"}

    summary = {
        "games_played": games,
        "matches": matches,
        "from_footer": {
            "total_gf": totals.get("goals"),
            "total_ga": totals.get("goals_against"),
            "total_shots": totals.get("shots"),
            "total_sot": totals.get("shots_on_target"),
        }
    }

    # Use footer when available, otherwise sum rows
    summary["avg_gf"] = round(totals.get("goals", sum(m["gf"] for m in matches)) / games, 2)
    summary["avg_ga"] = round(totals.get("goals_against", sum(m["ga"] for m in matches)) / games, 2)
    summary["avg_shots"] = round(totals.get("shots", sum(m["shots"] for m in matches)) / games, 1)
    summary["avg_sot"] = round(totals.get("shots_on_target", sum(m["sot"] for m in matches)) / games, 1)

    # Last 5 form (newest at bottom)
    last5 = matches[-5:]
    pts = sum(3 if "W" in m["result"] else 1 if "D" in m["result"] else 0 for m in last5)
    summary["form_last5_pts"] = pts
    summary["form_last5_ppg"] = round(pts / len(last5), 2) if last5 else 0

    return summary


# ── Example usage ─────────────────────────────────────────────────────
if __name__ == "__main__":
    team = "Aston Villa"
    slug = get_team_slug(team)
    if not slug:
        print(f"Could not find slug for {team}")
    else:
        data = parse_shooting_table(slug)
        print(json.dumps(data, indent=2))
        # You can also save to CSV/JSON like in your original cod
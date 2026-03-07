"""
diagnose_understat.py — Verify Understat endpoints work from your machine.

python diagnose_understat.py
"""
import json, time, urllib.request

def _get(url):
    time.sleep(1)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json, text/html",
        "Referer": "https://understat.com/",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode("utf-8"), r.status

# Understat has two useful endpoints:
# 1. Team season stats (JSON embedded in HTML)
# 2. Team match-by-match xG

tests = [
    ("Team page - Arsenal",     "https://understat.com/team/Arsenal/2025"),
    ("League page - EPL",       "https://understat.com/league/EPL/2025"),
    ("League page - La Liga",   "https://understat.com/league/La_liga/2025"),
    ("League page - Bundesliga","https://understat.com/league/Bundesliga/2025"),
    ("League page - Serie A",   "https://understat.com/league/Serie_A/2025"),
    ("League page - Ligue 1",   "https://understat.com/league/Ligue_1/2025"),
]

for label, url in tests:
    try:
        html, status = _get(url)
        has_xg    = "xG" in html or "x_g" in html or "xg" in html.lower()
        has_json  = "JSON.parse" in html or "var " in html
        has_cf    = "cloudflare" in html.lower() or "Just a moment" in html
        blocked   = "Access denied" in html or "403" in str(status)
        print(f"\n{'='*50}")
        print(f"{label}")
        print(f"  Status: {status}  HTML: {len(html)} chars")
        print(f"  ✓ Has xG data:    {has_xg}")
        print(f"  ✓ Has JSON embed: {has_json}")
        print(f"  ✗ Cloudflare:     {has_cf}")
        print(f"  ✗ Blocked:        {blocked}")
        # Show first JSON variable found
        if "var " in html:
            idx = html.find("var ")
            print(f"  First var: {html[idx:idx+80]}")
    except Exception as e:
        print(f"\n{label}: ERROR — {e}")

print("\nDone.")

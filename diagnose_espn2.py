"""
diagnose_espn2.py — Phase 2 diagnostic
Now that we know team IDs work, test stats + schedule endpoints.

Usage:
    python diagnose_espn2.py
"""
import json, time, urllib.request

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

def _get(url):
    time.sleep(0.8)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode()), None
    except Exception as e:
        return None, str(e)

def describe(obj, prefix="", depth=0):
    if depth > 5: return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                ln = f"[{len(v)}]" if isinstance(v, list) else ""
                print(f"{prefix}{k}: {type(v).__name__}{ln}")
                describe(v, prefix+"  ", depth+1)
            else:
                print(f"{prefix}{k}: {str(v)[:80]}")
    elif isinstance(obj, list):
        if not obj:
            print(f"{prefix}(empty)")
            return
        print(f"{prefix}[0] of {len(obj)}:")
        describe(obj[0], prefix+"  ", depth+1)

# eng.1 team IDs confirmed from previous run:
TEST_CASES = [
    ("eng.1", "349",  "AFC Bournemouth"),
    ("esp.1", "96",   "Alavés"),
    ("ger.1", "6418", "1. FC Heidenheim 1846"),
]

for league, team_id, team_name in TEST_CASES:
    print(f"\n{'='*60}")
    print(f"{league} / {team_name} (id={team_id})")
    print(f"{'='*60}")

    # --- Team Stats ---
    url = f"{ESPN_BASE}/{league}/teams/{team_id}/statistics"
    print(f"\n--- STATS: {url} ---")
    data, err = _get(url)
    if err or not data:
        print(f"  ERROR: {err}")
    else:
        describe(data)
        # Try to find any numeric stats
        flat = {}
        def flatten(obj, d=0):
            if d > 6: return
            if isinstance(obj, dict):
                n = obj.get("name","")
                v = obj.get("value")
                if n and v is not None:
                    flat[n] = v
                for val in obj.values():
                    flatten(val, d+1)
            elif isinstance(obj, list):
                for item in obj:
                    flatten(item, d+1)
        flatten(data)
        print(f"\n  → All stat name→value pairs found:")
        for k, v in list(flat.items())[:20]:
            print(f"     {k}: {v}")

    # --- Schedule ---
    url = f"{ESPN_BASE}/{league}/teams/{team_id}/schedule"
    print(f"\n--- SCHEDULE: {url} ---")
    data, err = _get(url)
    if err or not data:
        print(f"  ERROR: {err}")
    else:
        describe(data)
        # Find completed events
        events = data.get("events", [])
        completed = [e for e in events
                     if e.get("competitions",[{}])[0]
                          .get("status",{}).get("type",{})
                          .get("completed", False)]
        print(f"\n  → {len(events)} total events, {len(completed)} completed")
        if completed:
            last = completed[-1]
            print(f"  → Last completed: {last.get('date','')} {last.get('name','')}")

print("\nDone.")

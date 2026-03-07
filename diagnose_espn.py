"""
diagnose_espn.py — Run this on your Windows machine to see exactly what
ESPN returns for standings and teams endpoints.

Usage:
    python diagnose_espn.py eng.1
    python diagnose_espn.py eng.1 esp.1 ger.1
"""
import json, sys, time, urllib.request

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

def _get(url):
    time.sleep(0.8)
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode())

def describe(obj, prefix="", depth=0):
    """Recursively describe the structure of a JSON object."""
    if depth > 6:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                length = len(v) if isinstance(v, list) else ""
                print(f"{prefix}{k}: {type(v).__name__}{f'[{length}]' if length != '' else ''}")
                describe(v, prefix + "  ", depth + 1)
            else:
                val = str(v)[:60] if v is not None else "null"
                print(f"{prefix}{k}: {val}")
    elif isinstance(obj, list):
        if len(obj) == 0:
            print(f"{prefix}(empty list)")
            return
        print(f"{prefix}[0] of {len(obj)}:")
        describe(obj[0], prefix + "  ", depth + 1)

codes = sys.argv[1:] if len(sys.argv) > 1 else ["eng.1"]

for code in codes:
    print(f"\n{'='*60}")
    print(f"LEAGUE: {code}")
    print(f"{'='*60}")

    # --- Standings ---
    print(f"\n--- STANDINGS: {ESPN_BASE}/{code}/standings ---")
    try:
        data = _get(f"{ESPN_BASE}/{code}/standings")
        describe(data)

        # Find anything with 'team' + 'stats'
        found = []
        def hunt(obj, path=""):
            if isinstance(obj, dict):
                if "team" in obj and "stats" in obj:
                    found.append((path, obj))
                    return
                for k, v in obj.items():
                    hunt(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    hunt(item, f"{path}[{i}]")
        hunt(data)
        print(f"\n  → Found {len(found)} team+stats entries")
        if found:
            path, entry = found[0]
            print(f"  → First entry at: {path}")
            print(f"  → team.displayName: {entry.get('team',{}).get('displayName','?')}")
            stat_names = [s.get('name') for s in entry.get('stats',[])[:8]]
            print(f"  → stat names: {stat_names}")
        else:
            print("  !! NO entries found — dumping top 400 chars of raw JSON:")
            print(json.dumps(data, indent=2)[:400])
    except Exception as e:
        print(f"  ERROR: {e}")

    # --- Teams ---
    print(f"\n--- TEAMS: {ESPN_BASE}/{code}/teams ---")
    try:
        data = _get(f"{ESPN_BASE}/{code}/teams")
        describe(data)
        # Try to find team list
        teams = (data.get("sports",[{}])[0]
                    .get("leagues",[{}])[0]
                    .get("teams",[]))
        print(f"\n  → Found {len(teams)} teams via sports[0].leagues[0].teams")
        if teams:
            t = teams[0].get("team", teams[0])
            print(f"  → First team: id={t.get('id')} name={t.get('displayName')}")
        elif not teams:
            # Try alternate path
            print("  → Trying alternate paths...")
            if "items" in data:
                print(f"  → data.items: {len(data['items'])}")
            elif "teams" in data:
                print(f"  → data.teams: {len(data['teams'])}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\nDone. Paste the output back so we can fix the parser.")

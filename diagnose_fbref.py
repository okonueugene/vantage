"""
diagnose_fbref.py — See exactly what FBref returns for Arsenal.
Saves the raw HTML so we can inspect what's blocking the selector.

python diagnose_fbref.py
"""
import time, sys
from playwright.sync_api import sync_playwright

URL = "https://fbref.com/en/squads/18bb7c10/2025-2026/matchlogs/all_comps/shooting/"

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=False,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox"]
    )
    ctx = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        locale="en-GB",
    )
    page = ctx.new_page()
    page.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    print(f"Loading: {URL}")
    try:
        page.goto(URL, wait_until="domcontentloaded", timeout=30000)
    except Exception as e:
        print(f"goto error: {e}")

    # Wait a bit and check what's on the page
    time.sleep(3)
    title = page.title()
    url_now = page.url
    print(f"Title: {title}")
    print(f"URL:   {url_now}")

    html = page.content()
    print(f"HTML length: {len(html)}")

    # Look for key indicators
    checks = [
        ("#matchlogs_for",        "Main table found"),
        ("matchlogs_for",         "Table ID in HTML"),
        ("consent",               "Consent/cookie wall"),
        ("captcha",               "CAPTCHA"),
        ("cloudflare",            "Cloudflare block"),
        ("Just a moment",         "CF challenge page"),
        ("Access denied",         "Access denied"),
        ("Too Many Requests",     "Rate limited (429)"),
        ("Are you a robot",       "Robot check"),
        ("Enable JavaScript",     "JS required page"),
        ("shooting",              "Word 'shooting' in page"),
        ("Arsenal",               "Arsenal mentioned"),
    ]
    print("\nPage content checks:")
    for selector, label in checks:
        found = selector.lower() in html.lower()
        print(f"  {'✓' if found else '✗'} {label}")

    # Save HTML for inspection
    with open("fbref_debug.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("\nFull HTML saved to fbref_debug.html")

    # Check all table IDs on the page
    tables = page.query_selector_all("table")
    print(f"\nTables found on page: {len(tables)}")
    for t in tables[:10]:
        tid = t.get_attribute("id") or "(no id)"
        print(f"  table id={tid}")

    # Check for any consent buttons
    btns = page.query_selector_all("button")
    print(f"\nButtons: {len(btns)}")
    for b in btns[:5]:
        txt = b.inner_text()[:50] if b.inner_text() else ""
        print(f"  button: {txt}")

    browser.close()
print("\nDone.")

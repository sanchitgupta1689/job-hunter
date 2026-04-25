"""
Instahyre browser helpers — session management and LLM-powered job scraping.
"""

import asyncio
import json
from pathlib import Path
from playwright.async_api import Browser, BrowserContext, Page
from langchain_core.messages import HumanMessage, SystemMessage

SESSIONS_DIR      = Path(__file__).parent.parent / "sessions"
INSTAHYRE_SESSION = SESSIONS_DIR / "instahyre_session.json"
INSTAHYRE_BASE    = "https://www.instahyre.com"
MAX_PAGE_CHARS    = 6000


# ── Session helpers ────────────────────────────────────────────────────────────

async def load_context(browser: Browser) -> BrowserContext:
    if INSTAHYRE_SESSION.exists():
        return await browser.new_context(storage_state=str(INSTAHYRE_SESSION))
    return await browser.new_context()


async def save_session(context: BrowserContext) -> None:
    SESSIONS_DIR.mkdir(exist_ok=True)
    await context.storage_state(path=str(INSTAHYRE_SESSION))


async def _goto(page: Page, url: str, timeout: int = 20000) -> None:
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    await asyncio.sleep(2)


async def is_logged_in(page: Page) -> bool:
    try:
        await _goto(page, f"{INSTAHYRE_BASE}/candidate/opportunities/", timeout=20000)
        url = page.url
        return "login" not in url and "signin" not in url
    except Exception:
        return False


async def manual_login(page: Page) -> None:
    await _goto(page, f"{INSTAHYRE_BASE}/login/")
    print("\n" + "=" * 60)
    print("[Instahyre] Browser is open. Please log in manually.")
    print("[Instahyre] Wait until your dashboard loads, THEN press Enter.")
    print("=" * 60 + "\n")
    input("Press Enter once you are logged in to Instahyre: ")
    await asyncio.sleep(2)
    print("[Instahyre] Session captured.\n")


# ── LLM extraction ─────────────────────────────────────────────────────────────

JOBS_SYSTEM = """You are an Instahyre job listing parser.
Given raw text scraped from an Instahyre opportunities/jobs page, extract ALL visible job listings.
Return ONLY a JSON array:
[
  {
    "title": "job title",
    "company": "company name",
    "location": "city / remote",
    "salary": "salary range if shown, else empty string"
  }
]
Extract as many jobs as you can find."""


async def _page_text(page: Page) -> str:
    try:
        return (await page.inner_text("body"))[:MAX_PAGE_CHARS]
    except Exception:
        return ""


async def scrape_jobs(page: Page, llm, preferences: dict, limit: int = 30) -> list[dict]:
    """
    Scrape job recommendations from Instahyre using LLM-powered extraction.
    """
    await _goto(page, f"{INSTAHYRE_BASE}/candidate/opportunities/", timeout=20000)
    await asyncio.sleep(1)

    # Scroll to load more
    prev_count = 0
    for _ in range(2):
        count = await page.locator("[class*='opportunity'], [class*='job']").count()
        if count == prev_count:
            break
        prev_count = count
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1)

    # Collect URLs from DOM
    url_map: dict[str, str] = {}
    links = await page.locator("a[href*='/job'], a[href*='/opportunity']").all()
    for link in links[:limit]:
        try:
            href  = await link.get_attribute("href", timeout=2000)
            label = await link.inner_text(timeout=2000)
            if href:
                full = f"{INSTAHYRE_BASE}{href}" if href.startswith("/") else href
                url_map[label.strip()[:60]] = full
        except Exception:
            continue

    # LLM extraction
    page_text = await _page_text(page)
    response  = await llm.ainvoke([
        SystemMessage(content=JOBS_SYSTEM),
        HumanMessage(content=f"Instahyre opportunities page text:\n{page_text}"),
    ])
    content = response.content
    if isinstance(content, list):
        raw = "\n".join(b["text"] if isinstance(b, dict) else getattr(b, "text", str(b)) for b in content)
    else:
        raw = str(content)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        jobs = json.loads(raw.strip())
    except Exception:
        jobs = []

    for job in jobs:
        title_key  = job.get("title", "")[:60]
        job["url"]    = url_map.get(title_key, "")
        job["source"] = "Instahyre"

    return jobs[:limit]

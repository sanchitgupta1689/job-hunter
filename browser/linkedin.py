"""
LinkedIn browser helpers — session management, profile scraping, job scraping.
LLM-powered extraction: raw page text is passed to Gemini for structured parsing.
"""

import asyncio
import json
from pathlib import Path
from playwright.async_api import Browser, BrowserContext, Page
from langchain_core.messages import HumanMessage, SystemMessage

SESSIONS_DIR   = Path(__file__).parent.parent / "sessions"
LINKEDIN_SESSION = SESSIONS_DIR / "linkedin_session.json"
LINKEDIN_BASE  = "https://www.linkedin.com"

MAX_PAGE_CHARS = 6000  # keep within LLM context


# ── Session helpers ────────────────────────────────────────────────────────────

async def load_context(browser: Browser) -> BrowserContext:
    if LINKEDIN_SESSION.exists():
        return await browser.new_context(storage_state=str(LINKEDIN_SESSION))
    return await browser.new_context()


async def save_session(context: BrowserContext) -> None:
    SESSIONS_DIR.mkdir(exist_ok=True)
    await context.storage_state(path=str(LINKEDIN_SESSION))


async def _goto(page: Page, url: str, timeout: int = 30000) -> None:
    """Navigate and wait only for domcontentloaded — avoids LinkedIn's infinite network activity."""
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    await asyncio.sleep(2)


async def is_logged_in(page: Page) -> bool:
    try:
        await _goto(page, f"{LINKEDIN_BASE}/feed/", timeout=20000)
        url = page.url
        return "feed" in url or "mynetwork" in url
    except Exception:
        return False


async def manual_login(page: Page) -> None:
    await _goto(page, f"{LINKEDIN_BASE}/login")
    print("\n" + "=" * 60)
    print("[LinkedIn] Browser is open. Please log in manually.")
    print("[LinkedIn] Wait until your feed loads, THEN press Enter.")
    print("=" * 60 + "\n")
    input("Press Enter once you are logged in to LinkedIn: ")
    await asyncio.sleep(2)
    print("[LinkedIn] Session captured.\n")


# ── LLM extraction helpers ─────────────────────────────────────────────────────

async def _page_text(page: Page, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Extract main content text — avoids nav/footer bloat that inflates token count."""
    for selector in ["main", "#main", ".scaffold-layout__main", ".application-outlet", "article"]:
        try:
            text = await page.locator(selector).first.inner_text(timeout=2000)
            if len(text) > 200:
                return text[:max_chars]
        except Exception:
            pass
    try:
        return (await page.inner_text("body"))[:max_chars]
    except Exception:
        return ""


def _extract_text(content) -> str:
    """Normalize LLM response content — handles str or list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)
    return str(content)


async def _llm_extract(llm, system: str, content: str) -> dict | list:
    response = await llm.ainvoke([
        SystemMessage(content=system),
        HumanMessage(content=content),
    ])
    raw = _extract_text(response.content).strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return {}


# ── Profile scraping ───────────────────────────────────────────────────────────

PROFILE_SYSTEM = """You are a LinkedIn profile parser.
You are given concatenated text from 3 LinkedIn pages: main profile, skills page, experience page.
Extract the following and return ONLY valid JSON:
{
  "name": "full name",
  "headline": "current job title / headline",
  "location": "city, country",
  "about": "about section summary (max 200 chars)",
  "skills": ["skill1", "skill2", ...],
  "experience": [
    {"title": "Job Title", "company": "Company", "duration": "Jan 2020 - Present"},
    ...
  ]
}
Return at most 15 skills and 5 experience entries. If a field is missing use empty string or list."""


async def scrape_profile(page: Page, llm) -> dict:
    """
    Scrape LinkedIn profile with a single LLM call.
    Collects text from 3 pages, combines them, sends once to Gemini.
    """
    await _goto(page, f"{LINKEDIN_BASE}/in/me/", timeout=30000)
    await asyncio.sleep(1)

    actual_url = page.url
    print(f"[LinkedIn] Profile URL: {actual_url}")

    if "login" in actual_url or "authwall" in actual_url:
        print("[LinkedIn] Redirected to login — not authenticated")
        return {}

    sections: list[str] = []

    # Main profile page
    sections.append("=== MAIN PROFILE ===\n" + await _page_text(page, max_chars=2000))

    # Skills page
    try:
        await _goto(page, actual_url.rstrip("/") + "/details/skills/", timeout=20000)
        await asyncio.sleep(1)
        sections.append("=== SKILLS PAGE ===\n" + await _page_text(page, max_chars=2000))
    except Exception:
        pass

    # Experience page
    try:
        await _goto(page, actual_url.rstrip("/") + "/details/experience/", timeout=20000)
        await asyncio.sleep(1)
        sections.append("=== EXPERIENCE PAGE ===\n" + await _page_text(page, max_chars=2000))
    except Exception:
        pass

    combined = "\n\n".join(sections)
    profile = await _llm_extract(llm, PROFILE_SYSTEM, combined)
    if not isinstance(profile, dict):
        profile = {}
    profile["profile_url"] = actual_url
    return profile


# ── Jobs scraping ──────────────────────────────────────────────────────────────

JOBS_SYSTEM = """You are a LinkedIn job listing parser.
Given raw text scraped from a LinkedIn Jobs search page, extract ALL visible job listings.
Return ONLY a JSON array:
[
  {
    "title": "job title",
    "company": "company name",
    "location": "city / remote",
    "url": "full job URL if visible, else empty string"
  }
]
Extract as many jobs as you can find in the text."""


async def _collect_cards(page: Page, limit: int) -> dict[str, str]:
    """Scroll current page aggressively and collect {title: url} from job cards."""
    url_map: dict[str, str] = {}
    prev_count = 0
    for _ in range(6):
        cards = await page.locator(".job-card-container, [data-job-id]").all()
        if len(cards) >= limit or len(cards) == prev_count:
            break
        prev_count = len(cards)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(1)
        # Click "See more jobs" button if it appears
        try:
            btn = page.locator("button:has-text('See more jobs'), button:has-text('Show more')").first
            if await btn.is_visible(timeout=1000):
                await btn.click()
                await asyncio.sleep(1)
        except Exception:
            pass

    cards = await page.locator(".job-card-container, [data-job-id]").all()
    for card in cards[:limit]:
        try:
            href  = await card.locator("a").first.get_attribute("href", timeout=2000)
            label = await card.locator("a").first.inner_text(timeout=2000)
            if href:
                full = f"{LINKEDIN_BASE}{href}" if href.startswith("/") else href
                url_map[label.strip()[:60]] = full
        except Exception:
            continue
    return url_map


async def scrape_jobs(page: Page, llm, query: str, location: str = "", limit: int = 75) -> list[dict]:
    """
    Search LinkedIn Jobs across multiple result pages using the start= pagination parameter.
    Collects URLs from DOM, uses LLM to parse job text per page.
    """
    from urllib.parse import quote

    all_jobs: list[dict] = []
    url_map:  dict[str, str] = {}

    # LinkedIn returns ~25 jobs per page; paginate across 3 pages
    for start in range(0, 75, 25):
        if len(all_jobs) >= limit:
            break

        search_url = (
            f"{LINKEDIN_BASE}/jobs/search/"
            f"?keywords={quote(query)}"
            f"&location={quote(location)}"
            f"&f_TPR=r604800"
            f"&start={start}"
        )
        print(f"[LinkedIn] Scraping jobs page start={start} for '{query}'")
        await _goto(page, search_url, timeout=20000)
        await asyncio.sleep(1)

        # Collect URLs + scroll within page
        page_urls = await _collect_cards(page, limit=30)
        url_map.update(page_urls)

        # LLM extracts structured data from page text
        page_text = await _page_text(page, max_chars=5000)
        page_jobs = await _llm_extract(llm, JOBS_SYSTEM, f"LinkedIn Jobs search page text:\n{page_text}")
        if isinstance(page_jobs, list):
            all_jobs.extend(page_jobs)

        if not page_jobs:
            break  # no more results

    # Deduplicate by title+company
    seen:        set  = set()
    unique_jobs: list = []
    for job in all_jobs:
        key = f"{job.get('title','')}|{job.get('company','')}".lower()
        if key not in seen:
            seen.add(key)
            title_key    = job.get("title", "")[:60]
            job["url"]   = url_map.get(title_key, "")
            job["source"] = "LinkedIn"
            unique_jobs.append(job)

    return unique_jobs[:limit]

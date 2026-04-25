"""
LinkedIn Jobs Agent — searches LinkedIn Jobs using derived preferences
and returns a raw list of job listings.
"""

import json
from langsmith import traceable
from rich.console import Console

from browser.linkedin import scrape_jobs, is_logged_in, load_context, save_session

console = Console()


@traceable(name="LinkedIn Jobs Agent")
async def run_linkedin_agent(state: dict, browser, llm) -> dict:
    console.print("\n[bold cyan]🔍 LinkedIn Agent[/bold cyan] Searching LinkedIn Jobs...")

    preferences = state.get("preferences", {})
    titles      = preferences.get("preferred_titles", ["Software Engineer"])
    location    = preferences.get("preferred_location", "")

    ctx  = await load_context(browser)
    page = await ctx.new_page()

    if not await is_logged_in(page):
        console.print("   [yellow]⚠ LinkedIn session expired — please re-login[/yellow]")
        from browser.linkedin import manual_login
        await manual_login(page)
        await save_session(ctx)

    all_jobs = []
    # Search top 3 preferred titles
    for title in titles[:3]:
        console.print(f"   [dim]→ Searching: '{title}' in '{location}'[/dim]")
        jobs = await scrape_jobs(page, llm, title, location, limit=25)
        all_jobs.extend(jobs)

    await page.close()
    await ctx.close()

    # Deduplicate by URL
    seen = set()
    unique_jobs = []
    for job in all_jobs:
        key = job.get("url") or f"{job['title']}-{job['company']}"
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)

    console.print(f"   [dim]→ Found {len(unique_jobs)} unique LinkedIn jobs[/dim]")
    return {"linkedin_jobs": unique_jobs}

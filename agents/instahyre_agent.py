"""
Instahyre Agent — scrapes job recommendations from Instahyre
where the user's resume is already uploaded.
"""

import json
from langsmith import traceable
from rich.console import Console

from browser.instahyre import scrape_jobs, is_logged_in, manual_login, load_context, save_session

console = Console()


@traceable(name="Instahyre Agent")
async def run_instahyre_agent(state: dict, browser, llm) -> dict:
    console.print("\n[bold magenta]🏢 Instahyre Agent[/bold magenta] Fetching Instahyre matches...")

    preferences = state.get("preferences", {})

    ctx  = await load_context(browser)
    page = await ctx.new_page()

    logged_in = await is_logged_in(page)
    if not logged_in:
        await manual_login(page)
        await save_session(ctx)

    jobs = await scrape_jobs(page, llm, preferences, limit=30)

    await page.close()
    await ctx.close()

    console.print(f"   [dim]→ Found {len(jobs)} Instahyre matches[/dim]")
    return {"instahyre_jobs": jobs}

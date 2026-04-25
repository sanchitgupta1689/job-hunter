"""
Profile Agent — scrapes the user's LinkedIn profile using Playwright,
then uses Gemini to derive structured job preferences.
"""

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from rich.console import Console

from browser.linkedin import scrape_profile, is_logged_in, manual_login, load_context, save_session

console = Console()

SYSTEM_PROMPT = """You are a career analyst. Given a LinkedIn profile, extract structured job preferences.
Return ONLY valid JSON with these keys:
{
  "preferred_titles": ["list of 3-5 job titles to search for"],
  "skills": ["top 10 skills from the profile"],
  "preferred_location": "city or remote preference",
  "seniority": "junior/mid/senior/lead",
  "industries": ["2-3 preferred industries"],
  "summary": "2-sentence candidate summary"
}"""


@traceable(name="Profile Agent")
async def run_profile_agent(state: dict, llm, browser) -> dict:
    console.print("\n[bold blue]👤 Profile Agent[/bold blue] Extracting LinkedIn profile...")

    ctx  = await load_context(browser)
    page = await ctx.new_page()

    logged_in = await is_logged_in(page)
    if not logged_in:
        await manual_login(page)
        await save_session(ctx)

    raw_profile = await scrape_profile(page, llm)

    # Detect unauthenticated scrape (LinkedIn shows "Join LinkedIn" when not logged in)
    name = raw_profile.get("name", "")
    if not name or "join" in name.lower() or "linkedin" in name.lower():
        console.print("   [yellow]⚠ Not logged in — please log in again[/yellow]")
        await manual_login(page)
        await save_session(ctx)
        raw_profile = await scrape_profile(page, llm)

    await page.close()
    await ctx.close()

    console.print(f"   [dim]→ Profile scraped for: {raw_profile.get('name', 'Unknown')}[/dim]")

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"LinkedIn Profile:\n{json.dumps(raw_profile, indent=2)}"),
    ])

    try:
        c = response.content
        text = (c if isinstance(c, str) else "\n".join(b["text"] if isinstance(b, dict) else getattr(b, "text", str(b)) for b in c)).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        preferences = json.loads(text.strip())
    except Exception:
        preferences = {
            "preferred_titles": [raw_profile.get("headline", "Software Engineer")],
            "skills": raw_profile.get("skills", []),
            "preferred_location": raw_profile.get("location", ""),
            "seniority": "mid",
            "industries": [],
            "summary": raw_profile.get("about", "")[:200],
        }

    console.print(f"   [dim]→ Derived preferences: {preferences.get('preferred_titles', [])}[/dim]")

    return {
        "raw_profile":  raw_profile,
        "preferences":  preferences,
    }

"""
LangChain tools that wrap Playwright browser actions.
Each tool is async and uses a shared browser context passed at construction time.
"""

import json
from typing import Any
from langchain_core.tools import tool
from playwright.async_api import Page


def make_scrape_profile_tool(page: Page):
    @tool
    async def scrape_linkedin_profile(_: str = "") -> str:
        """Scrape the current user's LinkedIn profile. Returns JSON with name, headline, location, about, skills, experience."""
        from browser.linkedin import scrape_profile
        profile = await scrape_profile(page)
        return json.dumps(profile, indent=2)
    return scrape_linkedin_profile


def make_scrape_linkedin_jobs_tool(page: Page):
    @tool
    async def scrape_linkedin_jobs(query_and_location: str) -> str:
        """
        Search LinkedIn Jobs. Input format: 'job title | location' (e.g. 'Data Scientist | Bangalore').
        Returns JSON list of jobs with title, company, location, url.
        """
        from browser.linkedin import scrape_jobs
        parts = [p.strip() for p in query_and_location.split("|")]
        query    = parts[0] if len(parts) > 0 else query_and_location
        location = parts[1] if len(parts) > 1 else ""
        jobs = await scrape_jobs(page, query, location)
        return json.dumps(jobs, indent=2)
    return scrape_linkedin_jobs


def make_scrape_instahyre_jobs_tool(page: Page):
    @tool
    async def scrape_instahyre_jobs(preferences_json: str = "{}") -> str:
        """
        Scrape job recommendations from Instahyre.
        Input: JSON string of preferences (optional). Returns JSON list of jobs.
        """
        from browser.instahyre import scrape_jobs
        try:
            preferences = json.loads(preferences_json)
        except Exception:
            preferences = {}
        jobs = await scrape_jobs(page, preferences)
        return json.dumps(jobs, indent=2)
    return scrape_instahyre_jobs

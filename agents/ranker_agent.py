"""
Ranker Agent — uses Gemini to score and rank all scraped jobs
against the user's profile and preferences.
"""

import json
from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import traceable
from rich.console import Console

console = Console()

SYSTEM_PROMPT = """You are a senior recruiter and career advisor.
Given a candidate profile and a list of job listings, rank the top 15 most relevant jobs.

Return ONLY valid JSON — a list of objects with these keys:
{
  "rank": 1,
  "title": "job title",
  "company": "company name",
  "location": "city / remote",
  "url": "apply link",
  "source": "LinkedIn or Instahyre",
  "match_score": 85,
  "match_reason": "1-2 sentence explanation of why this is a good fit"
}

Sort by match_score descending. Be strict — only include genuinely relevant jobs."""


@traceable(name="Ranker Agent")
async def run_ranker_agent(state: dict, llm) -> dict:
    console.print("\n[bold yellow]⭐ Ranker Agent[/bold yellow] Scoring and ranking all jobs...")

    all_jobs = state.get("linkedin_jobs", []) + state.get("instahyre_jobs", [])
    preferences = state.get("preferences", {})
    raw_profile = state.get("raw_profile", {})

    if not all_jobs:
        console.print("   [yellow]⚠ No jobs to rank[/yellow]")
        return {"ranked_jobs": []}

    prompt = (
        f"Candidate Profile:\n{json.dumps(raw_profile, indent=2)}\n\n"
        f"Derived Preferences:\n{json.dumps(preferences, indent=2)}\n\n"
        f"Job Listings ({len(all_jobs)} total):\n{json.dumps(all_jobs, indent=2)}"
    )

    response = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    try:
        c = response.content
        text = (c if isinstance(c, str) else "\n".join(b["text"] if isinstance(b, dict) else getattr(b, "text", str(b)) for b in c)).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        ranked_jobs = json.loads(text.strip())
    except Exception as e:
        console.print(f"   [red]⚠ Ranking parse error: {e} — returning raw jobs[/red]")
        ranked_jobs = [
            {**job, "rank": i + 1, "match_score": 50, "match_reason": "Unranked"}
            for i, job in enumerate(all_jobs[:15])
        ]

    console.print(f"   [dim]→ Top job: {ranked_jobs[0]['title']} @ {ranked_jobs[0]['company']} (score: {ranked_jobs[0]['match_score']})[/dim]")
    return {"ranked_jobs": ranked_jobs}

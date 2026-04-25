"""
Report Agent — formats ranked jobs into a markdown report
and saves it via the MCP filesystem tool.
"""

import json
from datetime import datetime
from langsmith import traceable
from rich.console import Console

console = Console()


def _build_markdown(state: dict) -> str:
    preferences = state.get("preferences", {})
    ranked_jobs  = state.get("ranked_jobs", [])
    raw_profile  = state.get("raw_profile", {})
    now          = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Job Hunter Report",
        f"*Generated: {now}*",
        f"*Candidate: {raw_profile.get('name', 'Unknown')}*",
        "",
        "---",
        "",
        "## Candidate Summary",
        f"**Headline:** {raw_profile.get('headline', '')}",
        f"**Location:** {raw_profile.get('location', '')}",
        f"**Preferred Titles:** {', '.join(preferences.get('preferred_titles', []))}",
        f"**Key Skills:** {', '.join(preferences.get('skills', [])[:8])}",
        "",
        "---",
        "",
        f"## Top {len(ranked_jobs)} Job Recommendations",
        "",
    ]

    for job in ranked_jobs:
        lines += [
            f"### {job['rank']}. {job['title']} — {job['company']}",
            f"**Location:** {job['location']}  |  **Source:** {job['source']}  |  **Match Score:** {job.get('match_score', 'N/A')}/100",
            f"**Why:** {job.get('match_reason', '')}",
            f"**Apply:** {job.get('url', 'N/A')}",
            "",
        ]

    return "\n".join(lines)


@traceable(name="Report Agent")
async def run_report_agent(state: dict, mcp_tools: dict) -> dict:
    console.print("\n[bold green]📄 Report Agent[/bold green] Writing job report via MCP...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"job_report_{timestamp}"
    content   = _build_markdown(state)

    result = await mcp_tools["write_report"].ainvoke({
        "filename": filename,
        "content":  content,
    })

    report_path = f"output/{filename}.md"
    console.print(f"   [dim]→ Report saved: {report_path}[/dim]")

    return {"report_path": report_path}

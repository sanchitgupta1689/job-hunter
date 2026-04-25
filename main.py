"""
Job Hunter Agent — Entry Point
Multi-agent system: LinkedIn Profile → LinkedIn Jobs + Instahyre → Ranked Report
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv()

console = Console()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL          = "gemini-3-flash-preview"

# LangSmith tracing
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"]    = os.environ.get("LANGSMITH_PROJECT", "job-hunter")


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]🤖  Agentic Job Hunter[/bold cyan]\n"
        "[dim]Profile Agent → [LinkedIn ‖ Instahyre] → Ranker → Report[/dim]",
        border_style="cyan",
    ))


def print_results(state: dict):
    ranked_jobs = state.get("ranked_jobs", [])
    if not ranked_jobs:
        console.print("[yellow]No jobs ranked.[/yellow]")
        return

    table = Table(title="Top Job Recommendations", border_style="green", show_lines=True)
    table.add_column("#",           style="bold", width=3)
    table.add_column("Title",       style="cyan",  max_width=30)
    table.add_column("Company",     style="white", max_width=20)
    table.add_column("Location",    style="dim",   max_width=18)
    table.add_column("Score",       style="green", width=7)
    table.add_column("Source",      style="magenta", width=10)
    table.add_column("Why",         style="dim",   max_width=40)

    for job in ranked_jobs[:15]:
        table.add_row(
            str(job.get("rank", "")),
            job.get("title", ""),
            job.get("company", ""),
            job.get("location", ""),
            str(job.get("match_score", "")),
            job.get("source", ""),
            job.get("match_reason", "")[:80],
        )

    console.print()
    console.print(table)
    console.print(f"\n[dim]📄 Full report saved to: {state.get('report_path', 'output/')}[/dim]")


async def main():
    if not GEMINI_API_KEY:
        console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not set in .env")
        sys.exit(1)

    print_banner()
    console.print("[dim]Starting job hunt... A browser window will open for LinkedIn & Instahyre.[/dim]\n")

    try:
        from agents.orchestrator import run_job_hunter
        state = await run_job_hunter(GEMINI_API_KEY, MODEL)
        print_results(state)
    except KeyboardInterrupt:
        console.print("\n[bold]Interrupted.[/bold]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

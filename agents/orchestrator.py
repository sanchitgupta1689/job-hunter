"""
LangGraph Orchestrator — wires all agents into a multi-agent graph.

Flow:
  START
    ↓
  profile_agent          ← scrapes LinkedIn profile, derives preferences
    ↓          ↓
  linkedin_agent   instahyre_agent   ← run in PARALLEL
    ↓          ↓
  ranker_agent           ← scores all jobs against profile
    ↓
  report_agent           ← writes markdown report via MCP
    ↓
  email_agent            ← sends HTML email with results
    ↓
  END

Headless mode: auto-enabled when both session files exist (scheduled runs).
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agents.profile_agent   import run_profile_agent
from agents.linkedin_agent  import run_linkedin_agent
from agents.instahyre_agent import run_instahyre_agent
from agents.ranker_agent    import run_ranker_agent
from agents.report_agent    import run_report_agent
from agents.email_agent     import run_email_agent
from tools.mcp_client       import get_mcp_tools

SESSIONS_DIR = Path(__file__).parent.parent / "sessions"


# ── Shared State ───────────────────────────────────────────────────────────────

class JobHunterState(TypedDict):
    raw_profile:    dict
    preferences:    dict
    linkedin_jobs:  list
    instahyre_jobs: list
    ranked_jobs:    list
    report_path:    str


# ── Headless detection ─────────────────────────────────────────────────────────

def sessions_exist() -> bool:
    """Run headless if both session files are already saved."""
    li = (SESSIONS_DIR / "linkedin_session.json").exists()
    ih = (SESSIONS_DIR / "instahyre_session.json").exists()
    return li and ih


# ── Node factories ─────────────────────────────────────────────────────────────

def make_profile_node(llm, browser):
    async def profile_node(state: JobHunterState) -> dict:
        return await run_profile_agent(state, llm, browser)
    return profile_node


def make_linkedin_node(browser, llm):
    async def linkedin_node(state: JobHunterState) -> dict:
        return await run_linkedin_agent(state, browser, llm)
    return linkedin_node


def make_instahyre_node(browser, llm):
    async def instahyre_node(state: JobHunterState) -> dict:
        return await run_instahyre_agent(state, browser, llm)
    return instahyre_node


def make_ranker_node(llm):
    async def ranker_node(state: JobHunterState) -> dict:
        return await run_ranker_agent(state, llm)
    return ranker_node


def make_report_node(mcp_tools: dict):
    async def report_node(state: JobHunterState) -> dict:
        return await run_report_agent(state, mcp_tools)
    return report_node


def make_email_node():
    async def email_node(state: JobHunterState) -> dict:
        return await run_email_agent(state)
    return email_node


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph(llm, browser, mcp_tools: dict):
    graph = StateGraph(JobHunterState)

    graph.add_node("profile_agent",   make_profile_node(llm, browser))
    graph.add_node("linkedin_agent",  make_linkedin_node(browser, llm))
    graph.add_node("instahyre_agent", make_instahyre_node(browser, llm))
    graph.add_node("ranker_agent",    make_ranker_node(llm))
    graph.add_node("report_agent",    make_report_node(mcp_tools))
    graph.add_node("email_agent",     make_email_node())

    graph.add_edge(START,             "profile_agent")
    graph.add_edge("profile_agent",   "linkedin_agent")   # ─┐ parallel fan-out
    graph.add_edge("profile_agent",   "instahyre_agent")  # ─┘
    graph.add_edge("linkedin_agent",  "ranker_agent")     # ─┐ fan-in
    graph.add_edge("instahyre_agent", "ranker_agent")     # ─┘
    graph.add_edge("ranker_agent",    "report_agent")
    graph.add_edge("report_agent",    "email_agent")
    graph.add_edge("email_agent",     END)

    return graph.compile()


# ── Main runner ────────────────────────────────────────────────────────────────

SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mcp_server.py")
SERVER_PARAMS = StdioServerParameters(command=sys.executable, args=[SERVER_SCRIPT])


@traceable(name="Job Hunter Workflow")
async def run_job_hunter(gemini_api_key: str, model: str = "gemini-3-flash-preview") -> dict:
    from playwright.async_api import async_playwright

    headless = sessions_exist()

    llm = ChatGoogleGenerativeAI(model=model, google_api_key=gemini_api_key, temperature=0)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)

        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = get_mcp_tools(session)

                app   = build_graph(llm, browser, mcp_tools)
                state = await app.ainvoke(
                    {
                        "raw_profile":    {},
                        "preferences":    {},
                        "linkedin_jobs":  [],
                        "instahyre_jobs": [],
                        "ranked_jobs":    [],
                        "report_path":    "",
                    },
                    config={"run_name": "Job Hunter Run"},
                )

        await browser.close()

    return state

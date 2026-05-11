# Job Hunter

An agentic AI system that scouts relevant job opportunities based on your LinkedIn profile — fully automated, from profile extraction to a ranked daily report.

## Overview

Job Hunter uses browser automation and LLM-powered agents to:
1. Extract your professional profile from LinkedIn
2. Scrape fresh job listings from multiple portals
3. Rank jobs by relevance to your profile and interests
4. Deliver a daily curated job report

## How It Works

```
LinkedIn / Instahyre
        │
        ▼
┌─────────────────┐
│  Profile Agent  │  ← Logs in via Playwright, extracts profile details & interests
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Job Agent     │  ← Navigates job portals, scrapes relevant listings
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Ranking Engine  │  ← Scores jobs against your profile
└────────┬────────┘
         │
         ▼
   Daily Report
```

## Agents

### Profile Agent
- Logs into LinkedIn (and other portals) using Playwright browser automation
- Extracts key profile details: skills, experience, education, headline
- Captures stated interests and job preferences

### Job Agent
- Navigates to configured job portals (LinkedIn, Instahyre, etc.)
- Scrapes job listings using browser automation
- Normalizes job data across portals

### Ranking Engine
- Scores each job against your extracted profile
- Weighs factors like skill match, role relevance, and location
- Produces a sorted list of the most relevant opportunities

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM | [Google Gemini](https://ai.google.dev/) |
| Browser automation | [Playwright](https://playwright.dev/) |
| Observability | [LangSmith](https://smith.langchain.com/) |

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
playwright install
```

### Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
LINKEDIN_EMAIL=your_linkedin_email
LINKEDIN_PASSWORD=your_linkedin_password
```

### Run

```bash
python main.py
```

## Output

The agent generates a daily report (email / file) with:
- Job title, company, and location
- Relevance score and match reasoning
- Direct application link

## Observability

All agent runs are traced in LangSmith, giving full visibility into each step of the pipeline — profile extraction, job scraping, and ranking decisions.

## Notes

- Use a dedicated LinkedIn account or be mindful of LinkedIn's automation policies.
- Credentials are read from environment variables and never stored.

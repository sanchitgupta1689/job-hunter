"""
MCP server for the Job Hunter project.
Provides tools: write_report, read_report, list_reports.
Run as a subprocess via StdioServerParameters.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

server = Server("job-hunter-mcp")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="write_report",
            description="Write the final job recommendations report to a markdown file in the output directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename without extension, e.g. 'jobs_2024_01_01'"},
                    "content":  {"type": "string", "description": "Full markdown content of the report"},
                },
                "required": ["filename", "content"],
            },
        ),
        types.Tool(
            name="read_report",
            description="Read a previously saved job report.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "Filename with or without .md extension"},
                },
                "required": ["filename"],
            },
        ),
        types.Tool(
            name="list_reports",
            description="List all saved job reports.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "write_report":
        filename = arguments["filename"].replace(".md", "") + ".md"
        path = OUTPUT_DIR / filename
        path.write_text(arguments["content"], encoding="utf-8")
        return [types.TextContent(type="text", text=f"Report saved to {path}")]

    if name == "read_report":
        filename = arguments["filename"]
        if not filename.endswith(".md"):
            filename += ".md"
        path = OUTPUT_DIR / filename
        if not path.exists():
            return [types.TextContent(type="text", text=f"File not found: {filename}")]
        return [types.TextContent(type="text", text=path.read_text(encoding="utf-8"))]

    if name == "list_reports":
        files = sorted(OUTPUT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return [types.TextContent(type="text", text="No reports saved yet.")]
        listing = "\n".join(f.name for f in files)
        return [types.TextContent(type="text", text=listing)]

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

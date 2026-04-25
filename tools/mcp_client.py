"""
MCP client helper — wraps MCP tools as LangChain-callable tools.
Mirrors the pattern from the research-assistant project.
"""

from mcp import ClientSession
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class WriteReportInput(BaseModel):
    filename: str = Field(description="Filename without extension")
    content:  str = Field(description="Full markdown content of the report")


class ReadReportInput(BaseModel):
    filename: str = Field(description="Filename with or without .md extension")


class EmptyInput(BaseModel):
    pass


def get_mcp_tools(session: ClientSession) -> dict:
    """Return a dict of {tool_name: LangChain tool} for the job-hunter MCP server."""

    async def _write_report(filename: str, content: str) -> str:
        result = await session.call_tool("write_report", {"filename": filename, "content": content})
        return result.content[0].text if result.content else "done"

    async def _read_report(filename: str) -> str:
        result = await session.call_tool("read_report", {"filename": filename})
        return result.content[0].text if result.content else ""

    async def _list_reports() -> str:
        result = await session.call_tool("list_reports", {})
        return result.content[0].text if result.content else ""

    write_report = StructuredTool.from_function(
        coroutine=_write_report,
        name="write_report",
        description="Write the final job recommendations report to a markdown file.",
        args_schema=WriteReportInput,
    )
    read_report = StructuredTool.from_function(
        coroutine=_read_report,
        name="read_report",
        description="Read a previously saved job report.",
        args_schema=ReadReportInput,
    )
    list_reports = StructuredTool.from_function(
        coroutine=_list_reports,
        name="list_reports",
        description="List all saved job reports.",
        args_schema=EmptyInput,
    )

    return {
        "write_report": write_report,
        "read_report":  read_report,
        "list_reports": list_reports,
    }

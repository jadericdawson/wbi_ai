from __future__ import annotations
from mcp.server.fastmcp import FastMCP

def register_md_tools(mcp: FastMCP):

    @mcp.tool()
    def md_section(title: str, body: str) -> str:
        return f"# {title}\n\n{body}\n"

    @mcp.tool()
    def md_code(lang: str, code: str) -> str:
        return f"```{lang}\n{code}\n```\n"

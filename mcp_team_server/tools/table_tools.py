from __future__ import annotations
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

def register_table_tools(mcp: FastMCP):

    @mcp.tool()
    def to_markdown_table(rows: List[Dict[str, Any]]) -> str:
        if not rows: return ""
        headers = list(rows[0].keys())
        header = "| " + " | ".join(headers) + " |"
        sep = "| " + " | ".join("---" for _ in headers) + " |"
        lines = [header, sep]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
        return "\n".join(lines) + "\n"

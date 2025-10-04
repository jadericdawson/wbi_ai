from __future__ import annotations
import httpx
from mcp.server.fastmcp import FastMCP

def register_http_tools(mcp: FastMCP, allowlist: tuple[str, ...] | list[str]):
    ALLOWED = set(allowlist or ())

    @mcp.tool()
    async def http_get_json(url: str):
        try:
            host = url.split("/")[2]
        except Exception:
            return "invalid URL"
        if ALLOWED and host not in ALLOWED:
            return f"host '{host}' not in allowlist"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.get(url, headers={"Accept":"application/json"})
                r.raise_for_status()
                return r.json()
        except Exception as e:
            return f"fetch error: {e}"

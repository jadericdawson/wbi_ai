from __future__ import annotations
import json
from mcp.server.fastmcp import FastMCP
from ..scratchpad.store import ScratchpadStore

def register_cache_tools(mcp: FastMCP, store: ScratchpadStore):

    @mcp.tool()
    def cache_put(session_id: str, key: str, value: str) -> str:
        store.add_note(session_id, "cache", "context", "system", f"KV::{key}={value}")
        return "ok"

    @mcp.tool()
    def cache_find(session_id: str, key: str) -> str:
        for n in reversed(store.get_notes(session_id, "context")):
            if n["content"].startswith("KV::"):
                k, _, v = n["content"][4:].partition("=")
                if k == key: return v
        return ""

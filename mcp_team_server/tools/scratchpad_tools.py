from __future__ import annotations
from mcp.server.fastmcp import FastMCP
from ..scratchpad.store import ScratchpadStore

def register_scratchpad_tools(mcp: FastMCP, store: ScratchpadStore):
    @mcp.tool()
    def read_notes(session_id: str, section: str | None = None):
        return store.get_notes(session_id, section)

    @mcp.tool()
    def add_note(session_id: str, agent: str, section: str, role: str, content: str):
        store.add_note(session_id, agent, section, role, content)
        return "ok"

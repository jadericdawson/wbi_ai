from __future__ import annotations
from mcp.server.fastmcp import FastMCP
from ..scratchpad.store import ScratchpadStore

def register_aiprojects_tools(mcp: FastMCP, store: ScratchpadStore, project_client):

    @mcp.tool()
    def aiprojects_thread_init(session_id: str) -> str:
        th = project_client.agents.create_thread()
        store.add_note(session_id, "aiprojects", "context", "system", f"KV::thread_id={th.id}")
        return th.id

    @mcp.tool()
    def aiprojects_thread_id(session_id: str) -> str:
        for n in reversed(store.get_notes(session_id, "context")):
            if n["content"].startswith("KV::thread_id="):
                return n["content"].split("=",1)[1]
        return ""

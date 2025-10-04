from __future__ import annotations
import os, json, pathlib
from typing import Any
from mcp.server.fastmcp import FastMCP

def register_fs_tools(mcp: FastMCP, allow_dirs: list[str] | tuple[str, ...]):
    ALLOW = [str(pathlib.Path(p).resolve()) for p in (allow_dirs or [])]

    def _safe(path: str) -> str | None:
        abs_p = str(pathlib.Path(path).resolve())
        return abs_p if any(abs_p.startswith(a) for a in ALLOW) else None

    @mcp.tool()
    def fs_read_text(path: str, encoding: str = "utf-8") -> dict[str, Any]:
        sp = _safe(path)
        if not sp: return {"error": "path not allowed"}
        try:
            return {"path": sp, "text": pathlib.Path(sp).read_text(encoding=encoding)}
        except Exception as e:
            return {"error": str(e)}

    @mcp.tool()
    def fs_listdir(path: str) -> dict[str, Any]:
        sp = _safe(path)
        if not sp: return {"error":"path not allowed"}
        try:
            entries = [str(p) for p in pathlib.Path(sp).iterdir()]
            return {"path": sp, "entries": entries}
        except Exception as e:
            return {"error": str(e)}

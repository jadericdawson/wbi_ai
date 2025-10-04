from __future__ import annotations
import json
from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP

def register_schema_tools(mcp: FastMCP):

    @mcp.tool()
    def schema_sketch(json_list_text: str) -> dict[str, Any]:
        """Given JSON list of objects, infer a field->types map."""
        try:
            rows = json.loads(json_list_text)
            if not isinstance(rows, list): return {"error":"expect list"}
        except Exception as e:
            return {"error": f"parse: {e}"}

        sig: Dict[str, set] = {}
        for r in rows:
            if not isinstance(r, dict): continue
            for k, v in r.items():
                t = type(v).__name__
                sig.setdefault(k, set()).add(t)
        return {k: sorted(list(v)) for k, v in sig.items()}

    @mcp.tool()
    def schema_diff(a_text: str, b_text: str) -> dict[str, Any]:
        """Diff two field->types maps produced by schema_sketch."""
        try:
            a = json.loads(a_text); b = json.loads(b_text)
        except Exception as e:
            return {"error": f"parse: {e}"}
        added = [k for k in b.keys() if k not in a]
        removed = [k for k in a.keys() if k not in b]
        changed = [k for k in a.keys() if k in b and a[k] != b[k]]
        return {"added": added, "removed": removed, "changed": changed}

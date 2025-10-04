from __future__ import annotations
import json
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP

def register_json_tools(mcp: FastMCP):

    @mcp.tool()
    def json_parse(text: str) -> dict[str, Any]:
        try:
            return {"ok": True, "data": json.loads(text)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @mcp.tool()
    def json_validate(data_json: str, schema_json: str) -> dict[str, Any]:
        try:
            data = json.loads(data_json)
            schema = json.loads(schema_json)
        except Exception as e:
            return {"ok": False, "error": f"json parse: {e}"}

        # Minimal validator (no external lib): structure + required + type
        def _validate(d, s, path="$"):
            errs = []
            t = s.get("type")
            if t:
                types = {"object": dict, "array": list, "string": str, "number": (int,float), "boolean": bool, "null": type(None)}
                py = types.get(t)
                if py and not isinstance(d, py):
                    errs.append(f"{path}: expected {t}, got {type(d).__name__}")
            if isinstance(d, dict):
                req = s.get("required", [])
                for r in req:
                    if r not in d: errs.append(f"{path}.{r}: required")
                props = s.get("properties", {})
                for k, v in d.items():
                    if k in props: errs += _validate(v, props[k], f"{path}.{k}")
            if isinstance(d, list) and "items" in s:
                for i, v in enumerate(d):
                    errs += _validate(v, s["items"], f"{path}[{i}]")
            return errs

        errors = _validate(data, schema)
        return {"ok": len(errors)==0, "errors": errors}

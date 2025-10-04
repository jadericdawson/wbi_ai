from __future__ import annotations
import csv, io
from typing import Any, List, Dict
from mcp.server.fastmcp import FastMCP

def register_csv_tools(mcp: FastMCP):

    @mcp.tool()
    def csv_preview(csv_text: str, n: int = 10, delimiter: str = ",") -> dict[str, Any]:
        rows: List[Dict[str, str]] = []
        reader = csv.DictReader(io.StringIO(csv_text), delimiter=delimiter)
        for i, r in enumerate(reader):
            if i >= n: break
            rows.append(r)
        return {"headers": reader.fieldnames or [], "rows": rows}

    @mcp.tool()
    def csv_select_cols(csv_text: str, cols_csv: str, delimiter: str = ",") -> dict[str, Any]:
        cols = [c.strip() for c in cols_csv.split(",") if c.strip()]
        r = csv.DictReader(io.StringIO(csv_text), delimiter=delimiter)
        out = [{c: row.get(c, "") for c in cols} for row in r]
        return {"headers": cols, "rows": out}

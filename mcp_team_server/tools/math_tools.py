from __future__ import annotations
from mcp.server.fastmcp import FastMCP

def register_math_tools(mcp: FastMCP):

    @mcp.tool()
    def calc(expression: str) -> str:
        # Very conservative: digits + + - * / ( ) . % ** //
        allowed = "0123456789+-*/().% "
        if not all(c in allowed for c in expression.replace("**","").replace("//","")):
            return "error: disallowed character"
        try:
            return str(eval(expression, {"__builtins__": {}}, {}))
        except Exception as e:
            return f"error: {e}"

    @mcp.tool()
    def lbs_to_kg(pounds: float) -> float:
        return pounds * 0.45359237

    @mcp.tool()
    def kg_to_lbs(kg: float) -> float:
        return kg / 0.45359237

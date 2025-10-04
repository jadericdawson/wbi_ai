from __future__ import annotations
from typing import Any, List, Dict
from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase

def register_neo4j_tools(mcp: FastMCP, uri: str, user: str, password: str, allow: bool = False):
    if not allow: return  # opt-in only

    driver = GraphDatabase.driver(uri, auth=(user, password))

    @mcp.tool()
    def neo4j_query(cypher: str, params_json: str = "{}") -> dict:
        import json
        try:
            params = json.loads(params_json)
        except Exception as e:
            return {"error": f"params parse: {e}"}
        try:
            with driver.session() as s:
                res = s.run(cypher, **params)
                return {"records": [r.data() for r in res]}
        except Exception as e:
            return {"error": str(e)}

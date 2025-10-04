import os
import json
from typing import Any, Dict, List, Optional

from azure.cosmos import CosmosClient, exceptions, PartitionKey
from mcp.server.fastmcp import FastMCP, Tool

# -------------------------------------------------------------------
# CosmosDB Client Initialization (from your .env values)
# -------------------------------------------------------------------

COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")
COSMOS_DB = os.getenv("COSMOS_DB", "mcpdb")   # default db name if not set

if not COSMOS_ENDPOINT or not COSMOS_KEY:
    raise RuntimeError("❌ Missing COSMOS_ENDPOINT or COSMOS_KEY in environment.")

cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = cosmos_client.get_database_client(COSMOS_DB)

# -------------------------------------------------------------------
# Tool Implementations
# -------------------------------------------------------------------

def _get_container(container_name: str):
    """Helper: return Cosmos container client."""
    return database.get_container_client(container_name)

async def cosmos_query(container_name: str, query: str, parameters: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Run a SQL query against a CosmosDB container.
    Args:
        container_name: The container to query.
        query: SQL query string, e.g. "SELECT * FROM c WHERE c.id = @id"
        parameters: Optional list of {"name": "@id", "value": "123"} dicts.
    Returns:
        List of matching documents.
    """
    container = _get_container(container_name)
    items = container.query_items(query=query, parameters=parameters or [], enable_cross_partition_query=True)
    return [item for item in items]

async def cosmos_insert(container_name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """Insert a new document into a Cosmos container."""
    container = _get_container(container_name)
    return container.create_item(item)

async def cosmos_replace(container_name: str, item_id: str, partition_key: Any, new_item: Dict[str, Any]) -> Dict[str, Any]:
    """Replace an existing Cosmos item by id and partition key."""
    container = _get_container(container_name)
    return container.replace_item(item=item_id, body=new_item, partition_key=partition_key)

async def cosmos_delete(container_name: str, item_id: str, partition_key: Any) -> str:
    """Delete a document from Cosmos."""
    container = _get_container(container_name)
    container.delete_item(item=item_id, partition_key=partition_key)
    return f"Deleted item {item_id} from {container_name}"

# -------------------------------------------------------------------
# MCP Registration
# -------------------------------------------------------------------

def register_cosmos_tools(mcp: FastMCP):
    """
    Register CosmosDB tools into MCP.
    """
    mcp.add_tool(Tool("cosmos.query", cosmos_query,
                      desc="Run a SQL query against CosmosDB. Args: container_name, query, parameters(optional)"))
    mcp.add_tool(Tool("cosmos.insert", cosmos_insert,
                      desc="Insert a document into CosmosDB. Args: container_name, item(dict)"))
    mcp.add_tool(Tool("cosmos.replace", cosmos_replace,
                      desc="Replace a CosmosDB item. Args: container_name, id, partition_key, new_item(dict)"))
    mcp.add_tool(Tool("cosmos.delete", cosmos_delete,
                      desc="Delete a CosmosDB item. Args: container_name, id, partition_key"))

from __future__ import annotations
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from .server import mcp

starlette_app = mcp.streamable_http_app()
app = Starlette(routes=[])
app.mount("/", starlette_app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST","DELETE"],
    expose_headers=["Mcp-Session-Id"],
)

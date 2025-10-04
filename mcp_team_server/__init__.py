# mcp_team_server/__init__.py
# keep lightweight; do not import submodules that have deps
from .scratchpad import ScratchpadStore  # convenience re-export
__all__ = ["ScratchpadStore"]

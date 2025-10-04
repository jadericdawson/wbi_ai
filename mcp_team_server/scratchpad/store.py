# mcp_team_server/scratchpad/store.py
from __future__ import annotations
import sqlite3
import json
import time
import uuid
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY,
  created_at REAL NOT NULL,
  task TEXT NOT NULL,
  inclusion_policy TEXT NOT NULL DEFAULT 'all',
  metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS notes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  ts REAL NOT NULL,
  agent TEXT NOT NULL,
  section TEXT NOT NULL,     -- e.g., 'plan','actor','critic','final_draft','context'
  role TEXT NOT NULL,        -- 'system' | 'user' | 'assistant' | 'validator'
  content TEXT NOT NULL,
  tags TEXT NOT NULL DEFAULT '[]',
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
CREATE TABLE IF NOT EXISTS finals(
  session_id TEXT PRIMARY KEY,
  ts REAL NOT NULL,
  content TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id)
);
"""

class ScratchpadStore:
    """
    Lightweight SQLite scratchpad for MCP sessions.
    - sessions: high-level session metadata
    - notes:    chronological notes per section
    - finals:   assembled final text (one per session)
    """

    def __init__(self, path: str):
        self.path = path
        self._init_db()

    # ---- DB setup ----
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as c:
            c.executescript(SCHEMA)

    # ---- Sessions ----
    def create_session(self, task: str, inclusion_policy: str = "all",
                       metadata: dict[str, Any] | None = None) -> str:
        sid = str(uuid.uuid4())
        with self._connect() as c:
            c.execute(
                "INSERT INTO sessions(id, created_at, task, inclusion_policy, metadata) "
                "VALUES (?,?,?,?,?)",
                (sid, time.time(), task, inclusion_policy, json.dumps(metadata or {})),
            )
        return sid

    def get_session(self, sid: str) -> dict[str, Any] | None:
        with self._connect() as c:
            row = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as c:
            rows = c.execute("SELECT * FROM sessions ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ---- Notes ----
    def add_note(self, sid: str, agent: str, section: str, role: str,
                 content: str, tags: Iterable[str] = ()):
        with self._connect() as c:
            c.execute(
                "INSERT INTO notes(session_id, ts, agent, section, role, content, tags) "
                "VALUES (?,?,?,?,?,?,?)",
                (sid, time.time(), agent, section, role, content, json.dumps(list(tags))),
            )

    def get_notes(self, sid: str, section: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as c:
            if section:
                rows = c.execute(
                    "SELECT * FROM notes WHERE session_id=? AND section=? ORDER BY ts ASC",
                    (sid, section),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM notes WHERE session_id=? ORDER BY ts ASC",
                    (sid,),
                ).fetchall()
        return [dict(r) for r in rows]

    # ---- Final ----
    def set_final(self, sid: str, content: str):
        with self._connect() as c:
            c.execute(
                "INSERT OR REPLACE INTO finals(session_id, ts, content) VALUES (?,?,?)",
                (sid, time.time(), content),
            )

    def get_final(self, sid: str) -> str | None:
        with self._connect() as c:
            row = c.execute("SELECT content FROM finals WHERE session_id=?", (sid,)).fetchone()
        return row["content"] if row else None

    # ---- Export ----
    def export_markdown(self, sid: str) -> str:
        sess = self.get_session(sid) or {}
        lines = [f"# Session {sid}", f"Task: {sess.get('task','')}", ""]
        for n in self.get_notes(sid):
            lines.append(f"## [{n['section']}] {n['agent']} ({n['role']})")
            lines.append(n["content"])
            lines.append("")
        final = self.get_final(sid)
        if final:
            lines.append("## [final] Assembled output")
            lines.append(final)
        return "\n".join(lines)

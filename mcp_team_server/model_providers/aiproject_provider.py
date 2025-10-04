from __future__ import annotations
from .base import ModelProvider
import time

class AIProjectProvider(ModelProvider):
    def __init__(self, project_client, agent_id: str):
        self.client = project_client
        self.agent_id = agent_id

    async def complete(self, system: str, prompt: str) -> str:
        thread = self.client.agents.create_thread()
        try:
            merged = f"[SYSTEM]\n{system or ''}\n\n[USER]\n{prompt}"
            self.client.agents.create_message(thread_id=thread.id, role="user", content=merged)
            run = self.client.agents.create_run(thread_id=thread.id, agent_id=self.agent_id)

            for _ in range(120):  # poll up to ~60s
                r = self.client.agents.get_run(thread_id=thread.id, run_id=run.id)
                if r.status in ("completed", "failed", "cancelled", "expired"):
                    break
                time.sleep(0.5)

            if r.status != "completed":
                return f"[AIProjects agent status: {r.status}]"

            msgs = self.client.agents.list_messages(thread_id=thread.id)
            content = ""
            for m in reversed(list(msgs)):
                if getattr(m, "role", "") == "assistant":
                    if hasattr(m, "content") and isinstance(m.content, list):
                        content = "".join(getattr(p, "text", "") or "" for p in m.content)
                    else:
                        content = getattr(m, "content", "") or ""
                    if content:
                        break
            return content or ""
        finally:
            try:
                self.client.agents.delete_thread(thread_id=thread.id)
            except Exception:
                pass

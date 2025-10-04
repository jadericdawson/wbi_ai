from __future__ import annotations
from .base import LLMToolAgent, AgentIO

ACTOR_SYS = """You are ActorAgent.
Write ONE section ONLY. If structured data are available, include them verbatim (no summaries)."""

class ActorAgent(LLMToolAgent):
    async def draft(self, section_id: str, objective: str, context: str) -> str:
        p = f"Section: {section_id}\nObjective: {objective}\nContext:\n{context}\n\nWrite this section only."
        return await self.run(AgentIO(system=ACTOR_SYS, prompt=p))

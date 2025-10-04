from __future__ import annotations
from .base import LLMToolAgent, AgentIO

RESEARCHER_SYS = """You are ResearcherAgent.
Devise targeted queries and specific endpoints to call via tools. Prefer primary sources.
Return JSON ONLY: {"queries":[{"label":"...","url":"...","notes":"..."}]}"""

class ResearcherAgent(LLMToolAgent):
    async def plan_research(self, topic: str) -> str:
        return await self.run(AgentIO(system=RESEARCHER_SYS, prompt=topic))

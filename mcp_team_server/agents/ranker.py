from __future__ import annotations
from .base import LLMToolAgent, AgentIO

RANKER_SYS = """You are RankerAgent.
Rank candidate passages/items by direct relevance to the task. Return top-k with rationales.
JSON ONLY: {"top":[{"id":"...","score":0..1,"why":"..."}]}"""

class RankerAgent(LLMToolAgent):
    async def rank(self, task: str, candidates: str) -> str:
        return await self.run(AgentIO(system=RANKER_SYS, prompt=f"Task:\n{task}\nCandidates:\n{candidates}"))

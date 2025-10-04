from __future__ import annotations
from .base import LLMToolAgent, AgentIO

DEDUP_SYS = """You are DeduplicatorAgent.
Remove true duplicates while preserving distinct rows. Explain the rule used. JSON ONLY:
{"kept":[...], "removed":[...], "rule":"..."}"""

class DeduplicatorAgent(LLMToolAgent):
    async def dedupe(self, json_records: str) -> str:
        return await self.run(AgentIO(system=DEDUP_SYS, prompt=json_records))

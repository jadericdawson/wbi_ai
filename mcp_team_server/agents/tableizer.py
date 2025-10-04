from __future__ import annotations
from .base import LLMToolAgent, AgentIO

TABLEIZER_SYS = """You are TableizerAgent.
Transform records into a markdown table without summarizing or omitting rows or fields."""

class TableizerAgent(LLMToolAgent):
    async def tableize(self, json_records: str) -> str:
        return await self.run(AgentIO(system=TABLEIZER_SYS, prompt=json_records))

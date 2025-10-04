from __future__ import annotations
from .base import LLMToolAgent, AgentIO

RETRIEVER_SYS = """You are RetrieverAgent.
Given a data need, choose appropriate registered tools and exact parameters.
Return JSON ONLY: {"calls":[{"tool":"http_get_json|fs_read_text|csv_preview|neo4j_query|...","params":{}}]}"""

class RetrieverAgent(LLMToolAgent):
    async def plan_calls(self, need: str) -> str:
        return await self.run(AgentIO(system=RETRIEVER_SYS, prompt=need))

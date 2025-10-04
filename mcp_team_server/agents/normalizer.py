from __future__ import annotations
from .base import LLMToolAgent, AgentIO

NORMALIZER_SYS = """You are NormalizerAgent.
Normalize heterogeneous fields and units; preserve all information. JSON IN ⇒ JSON OUT, same length if possible."""

class NormalizerAgent(LLMToolAgent):
    async def normalize(self, json_blob: str, policy: str = "") -> str:
        return await self.run(AgentIO(system=NORMALIZER_SYS, prompt=f"Policy:\n{policy}\nData:\n{json_blob}"))

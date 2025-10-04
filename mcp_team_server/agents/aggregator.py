from __future__ import annotations
from .base import LLMToolAgent, AgentIO

AGGREGATOR_SYS = """You are AggregatorAgent.
Append accepted sections verbatim under a new heading. Do not rewrite content."""

class AggregatorAgent(LLMToolAgent):
    async def assemble(self, current_final: str, accepted_section: str) -> str:
        p = f"Current final:\n{current_final}\n\nAppend verbatim:\n{accepted_section}"
        return await self.run(AgentIO(system=AGGREGATOR_SYS, prompt=p))

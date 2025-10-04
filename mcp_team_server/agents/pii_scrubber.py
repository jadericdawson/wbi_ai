from __future__ import annotations
from .base import LLMToolAgent, AgentIO

PII_SYS = """You are PIIScrubberAgent.
Redact PII (names, emails, phones, addresses, SSN) with bracketed tokens [REDACTED:TYPE]. Preserve structure."""

class PIIScrubberAgent(LLMToolAgent):
    async def scrub(self, text: str) -> str:
        return await self.run(AgentIO(system=PII_SYS, prompt=text))

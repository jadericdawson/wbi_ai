from __future__ import annotations
from .base import LLMToolAgent, AgentIO

SEC_SYS = """You are SecurityValidatorAgent.
Check for security issues: secrets, keys, URLs, credentials, injections. Return JSON ONLY:
{"status":"ok|warn|fail","findings":["..."],"remediation":["..."]}"""

class SecurityValidatorAgent(LLMToolAgent):
    async def validate(self, text: str) -> str:
        return await self.run(AgentIO(system=SEC_SYS, prompt=text))

from __future__ import annotations
from .base import LLMToolAgent, AgentIO

CRITIC_SYS = """You are CriticAgent.
Judge ONLY against acceptance criteria.
Return JSON ONLY: {"decision":"accept"|"revise","reasons":["..."],"revision_instructions":"..."}"""

class CriticAgent(LLMToolAgent):
    async def review(self, section_id: str, acceptance: str, draft: str) -> str:
        p = f"Section: {section_id}\nAcceptance:\n{acceptance}\nDraft:\n{draft}\nReview now."
        return await self.run(AgentIO(system=CRITIC_SYS, prompt=p))

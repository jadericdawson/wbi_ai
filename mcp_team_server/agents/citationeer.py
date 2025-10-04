from __future__ import annotations
from .base import LLMToolAgent, AgentIO

CITE_SYS = """You are CitationeerAgent.
Attach inline citations to each claim with source IDs from provided evidence. Return markdown with citations like [S1],[S2]."""

class CitationeerAgent(LLMToolAgent):
    async def cite(self, draft_md: str, evidence_md: str) -> str:
        p = f"Draft:\n{draft_md}\n\nEvidence:\n{evidence_md}\nAdd citations."
        return await self.run(AgentIO(system=CITE_SYS, prompt=p))

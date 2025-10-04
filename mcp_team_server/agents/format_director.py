from __future__ annotations
from .base import LLMToolAgent, AgentIO

FORMAT_SYS = """You are FormatDirectorAgent.
Given accepted sections, produce final layout: headings, TOC, optional appendix. Do not change content—only arrange."""

class FormatDirectorAgent(LLMToolAgent):
    async def layout(self, accepted_sections_md: str) -> str:
        return await self.run(AgentIO(system=FORMAT_SYS, prompt=accepted_sections_md))

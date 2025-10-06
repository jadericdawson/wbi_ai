from __future__ import annotations
from .base import LLMToolAgent, AgentIO

SUPERVISOR_SYS = """You are SupervisorAgent. Your role is to determine if the user's request has been fully addressed.
Review the entire scratchpad, including the initial task and all subsequent actions and reports.

If the user's request is fully answered, respond with JSON ONLY: {"status":"finished"}
Otherwise, respond with JSON ONLY: {"status":"not finished"}"""

class SupervisorAgent(LLMToolAgent):
    async def review(self, task: str, scratchpad: str) -> str:
        p = f"Initial Task: {task}\n\nScratchpad History:\n{scratchpad}\n\nBased on all of this, is the task finished?"
        return await self.run(AgentIO(system=SUPERVISOR_SYS, prompt=p))
from __future__ import annotations
from .base import LLMToolAgent, AgentIO

PLANNER_SYS = """You are PlannerAgent.
Goal: break the task into atomic sections so no single agent outputs the full final answer.
Prefer inclusion over summarization. Label sections S1, S2...
Return JSON ONLY: {"sections":[{"id":"S1","title":"...","objective":"...","acceptance":"bullet list"}]}"""

class PlannerAgent(LLMToolAgent):
    async def plan(self, task: str) -> str:
        return await self.run(AgentIO(system=PLANNER_SYS, prompt=f"Task: {task}\nPlan now."))

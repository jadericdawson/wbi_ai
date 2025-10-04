from __future__ import annotations
from .base import LLMToolAgent, AgentIO

ROUTER_SYS = """You are RouterAgent.
Given a user task, choose an execution path and which agents/tools to call in order.
Return JSON ONLY:
{"plan":[{"step":"research|retrieve|draft|tableize|normalize|dedupe|cite|scrub|validate|aggregate","args":{...}}]}"""

class RouterAgent(LLMToolAgent):
    async def route(self, task: str) -> str:
        return await self.run(AgentIO(system=ROUTER_SYS, prompt=f"Task:\n{task}\nRoute now."))

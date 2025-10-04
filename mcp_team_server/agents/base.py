from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from ..model_providers.base import ModelProvider

@dataclass
class AgentIO:
    system: str
    prompt: str

class Agent(Protocol):
    name: str
    async def run(self, io: AgentIO) -> str: ...

class LLMToolAgent:
    def __init__(self, name: str, provider: ModelProvider):
        self.name = name
        self.provider = provider

    async def run(self, io: AgentIO) -> str:
        return await self.provider.complete(io.system, io.prompt)

from __future__ import annotations
from abc import ABC, abstractmethod

class ModelProvider(ABC):
    @abstractmethod
    async def complete(self, system: str, prompt: str) -> str: ...

from __future__ import annotations
from .base import ModelProvider

class GPT41Provider(ModelProvider):
    def __init__(self, azure_openai_client, deployment: str):
        self.client = azure_openai_client
        self.deployment = deployment

    async def complete(self, system: str, prompt: str) -> str:
        rsp = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return rsp.choices[0].message.content or ""

from __future__ import annotations
from typing import Any
from .base import ModelProvider

"""
DeepSeekProvider

Updated for the latest azure-ai-inference where message helpers like
UserMessage/SystemMessage are no longer exported. We pass messages as
plain dicts: {"role": "...", "content": "..."} and prefer `output_text`
on the response, with safe fallbacks for other shapes.
"""

class DeepSeekProvider(ModelProvider):
    def __init__(self, chat_client: Any, deployment: str):
        self.client = chat_client
        self.deployment = deployment

    async def complete(self, system: str, prompt: str) -> str:
        # NOTE: azure-ai-inference ChatCompletionsClient.complete now accepts dict messages.
        result = self.client.complete(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system or ""},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )

        # Preferred GA convenience property (if present)
        text = getattr(result, "output_text", None)
        if text:
            return text

        # Fallbacks for variant response shapes (older/newer SDKs)
        try:
            choices = getattr(result, "choices", None)
            if choices:
                msg = choices[0].message
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    # Some SDKs return a list of text parts
                    return "".join(getattr(p, "text", "") or str(p) for p in content)
                return content or ""
        except Exception:
            pass

        return ""

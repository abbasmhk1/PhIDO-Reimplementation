"""Client for any OpenAI-compatible /chat/completions endpoint (Ollama, vLLM, LM Studio, OpenAI...)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenAICompatClient:
    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", timeout: float = 300,
                 no_think: bool = False):
        self.no_think = no_think  # appends Qwen3's "/no_think" switch to skip the reasoning phase
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatClient":
        model = os.environ.get("PHIDO_LLM_MODEL")
        if not model:
            raise RuntimeError(
                "Set PHIDO_LLM_MODEL (and optionally PHIDO_LLM_BASE_URL, PHIDO_LLM_API_KEY). "
                "Default base URL is Ollama: http://localhost:11434/v1"
            )
        return cls(
            base_url=os.environ.get("PHIDO_LLM_BASE_URL", "http://localhost:11434/v1"),
            model=model,
            api_key=os.environ.get("PHIDO_LLM_API_KEY", "not-needed"),
            no_think=os.environ.get("PHIDO_NO_THINK", "") not in ("", "0", "false"),
        )

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        if self.no_think:
            user = user + "\n/no_think"
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            }
        ).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.URLError as e:
            raise RuntimeError(f"LLM endpoint {self.url} unreachable: {e}") from e
        return data["choices"][0]["message"]["content"]

"""Async multi-provider LLM chat client compatible with the original emergentintegrations API."""

from __future__ import annotations

import os
import asyncio
import logging
from dataclasses import dataclass, field
from typing import List

import httpx

logger = logging.getLogger("cropdoctor.llm")

GEMINI_MODEL = "gemini-flash-latest"

MODEL_ALIASES = {
    ("openai", "gpt-5.1"): "gpt-4o",
    ("openai", "gpt-5.2"): "gpt-4o",
    ("gemini", "gemini-3-flash-preview"): GEMINI_MODEL,
    ("gemini", "gemini"): GEMINI_MODEL,
    ("anthropic", "claude-sonnet-4-5-20250929"): "claude-sonnet-4-20250514",
}

_PROVIDER_ENV = {
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
}


@dataclass
class ImageContent:
    image_base64: str
    mime_type: str = "image/jpeg"


@dataclass
class UserMessage:
    text: str
    file_contents: List[ImageContent] = field(default_factory=list)


def llm_configured() -> bool:
    """True when at least one provider API key is available for local runs."""
    if any(os.environ.get(name, "").strip() for names in _PROVIDER_ENV.values() for name in names):
        return True
    key = os.environ.get("EMERGENT_LLM_KEY", "").strip()
    return bool(key) and not key.startswith("sk-emergent")


def _resolve_api_key(provider: str, fallback_key: str) -> str:
    for name in _PROVIDER_ENV.get(provider, []):
        val = os.environ.get(name, "").strip()
        if val:
            return val
    key = (fallback_key or "").strip()
    if key and not key.startswith("sk-emergent"):
        return key
    env_hint = _PROVIDER_ENV.get(provider, ["API_KEY"])[0]
    raise ValueError(
        f"No API key for provider '{provider}'. Set {env_hint} in backend/.env "
        "(Emergent universal keys only work on Emergent-hosted deployments)."
    )


class LlmChat:
    def __init__(self, api_key: str, session_id: str, system_message: str = ""):
        self._api_key = api_key
        self._session_id = session_id
        self._system_message = system_message
        self._provider = "openai"
        self._model = "gpt-4o"

    def with_model(self, provider: str, model_name: str) -> "LlmChat":
        self._provider = provider.lower()
        self._model = MODEL_ALIASES.get((self._provider, model_name), model_name)
        return self

    async def send_message(self, message: UserMessage) -> str:
        api_key = _resolve_api_key(self._provider, self._api_key)
        if self._provider == "openai":
            return await self._send_openai(api_key, message)
        if self._provider == "gemini":
            return await self._send_gemini(api_key, message)
        if self._provider == "anthropic":
            return await self._send_anthropic(api_key, message)
        raise ValueError(f"Unsupported LLM provider: {self._provider}")

    async def _send_openai(self, api_key: str, message: UserMessage) -> str:
        content: list = [{"type": "text", "text": message.text}]
        for img in message.file_contents:
            b64 = img.image_base64.split(",", 1)[-1]
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.mime_type};base64,{b64}"},
                }
            )
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system_message},
                {"role": "user", "content": content},
            ],
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def _send_gemini(self, api_key: str, message: UserMessage) -> str:
        parts: list = []
        if self._system_message:
            parts.append({"text": self._system_message})
        parts.append({"text": message.text})
        for img in message.file_contents:
            b64 = img.image_base64.split(",", 1)[-1]
            parts.append({"inline_data": {"mime_type": img.mime_type, "data": b64}})
        body = {"contents": [{"parts": parts}]}
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )
        
        # Retry logic for rate limits and temporary failures
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    r = await client.post(
                        url,
                        headers={
                            "Content-Type": "application/json",
                            "X-goog-api-key": api_key,
                        },
                        json=body,
                    )
                    
                    # Handle 503 (Service Unavailable) and 429 (Rate Limited) with retry
                    if r.status_code in (503, 429):
                        if attempt < max_retries - 1:
                            wait_time = base_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Gemini API {r.status_code}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    r.raise_for_status()
                    data = r.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                    
            except httpx.HTTPStatusError as e:
                error_text = e.response.text[:300]
                if e.response.status_code in (503, 429) and attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"Gemini API error {e.response.status_code}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                raise ValueError(f"Gemini API returned {e.response.status_code}: {error_text}")
            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"Gemini API request failed, retrying in {wait_time}s: {str(e)[:100]}")
                    await asyncio.sleep(wait_time)
                    continue
                raise ValueError(f"Failed to reach Gemini API: {str(e)}")
        
        raise ValueError("Gemini API: Maximum retries exceeded")

    async def _send_anthropic(self, api_key: str, message: UserMessage) -> str:
        content: list = [{"type": "text", "text": message.text}]
        for img in message.file_contents:
            b64 = img.image_base64.split(",", 1)[-1]
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": img.mime_type,
                        "data": b64,
                    },
                }
            )
        body = {
            "model": self._model,
            "max_tokens": 4096,
            "system": self._system_message,
            "messages": [{"role": "user", "content": content}],
        }
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
        r.raise_for_status()
        data = r.json()
        return "".join(block["text"] for block in data["content"] if block.get("type") == "text")

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from .config import Config, get_llm_api_key, get_llm_endpoint
from .log import Logger
from .utils import cache_it, hash64, time_it


class LLMBackend(ABC):
    """Abstract base class for LLM API backends."""

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        gen_conf: dict[str, Any],
    ) -> str | None: ...

    @abstractmethod
    async def vision_chat(
        self,
        model: str,
        prompt: str,
        image_b64: str,
        gen_conf: dict[str, Any],
    ) -> str | None: ...


class OllamaBackend(LLMBackend):
    _OPTION_KEY_MAP = {
        "temperature": "temperature",
        "max_tokens": "num_predict",
        "top_p": "top_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "repeat_penalty": "repeat_penalty",
        "num_ctx": "num_ctx",
    }

    def __init__(self, host: str, timeout: int = 15 * 60):
        from ollama import AsyncClient as OllamaAsyncClient

        self._client = OllamaAsyncClient(host=host, timeout=timeout)

    def _build_options(self, gen_conf: dict[str, Any]) -> dict[str, Any]:
        return {
            v: gen_conf[k] for k, v in self._OPTION_KEY_MAP.items() if k in gen_conf
        }

    @staticmethod
    def _log_token_stats(resp):
        if not resp:
            return
        eval_count = resp.get("eval_count", 0)
        eval_duration_ns = resp.get("eval_duration", 0)
        prompt_eval_count = resp.get("prompt_eval_count", 0)
        total_duration_ns = resp.get("total_duration", 0)
        if eval_count > 0 and eval_duration_ns > 0:
            gen_tps = eval_count / (eval_duration_ns / 1e9)
            total_s = total_duration_ns / 1e9 if total_duration_ns else 0
            Logger.info(
                f"Token stats: prompt={prompt_eval_count}, completion={eval_count}, "
                f"speed={gen_tps:.1f} tok/s, total={total_s:.1f}s"
            )

    async def chat(self, model, messages, gen_conf):
        resp = await self._client.chat(
            model=model,
            messages=messages,
            options=self._build_options(gen_conf),
            keep_alive=10,
        )
        self._log_token_stats(resp)
        return resp["message"]["content"] if resp else None

    async def vision_chat(self, model, prompt, image_b64, gen_conf):
        resp = await self._client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
            options=self._build_options(gen_conf),
        )
        self._log_token_stats(resp)
        return resp["message"]["content"] if resp else None


class OpenAIBackend(LLMBackend):
    def __init__(self, api_key: str, base_url: str = "", timeout: int = 15 * 60):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required for OpenAI backend. "
                "Install with: pip install openai"
            ) from exc

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
        )

    @staticmethod
    def _build_params(gen_conf: dict[str, Any]) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if "temperature" in gen_conf:
            params["temperature"] = gen_conf["temperature"]
        if "top_p" in gen_conf:
            params["top_p"] = gen_conf["top_p"]
        if "max_tokens" in gen_conf:
            params["max_tokens"] = gen_conf["max_tokens"]
        if "num_ctx" in gen_conf:
            params.setdefault("max_tokens", gen_conf["num_ctx"])
        if "presence_penalty" in gen_conf:
            params["presence_penalty"] = gen_conf["presence_penalty"]
        if "frequency_penalty" in gen_conf:
            params["frequency_penalty"] = gen_conf["frequency_penalty"]
        return params

    @staticmethod
    def _log_token_stats(resp, elapsed: float):
        usage = getattr(resp, "usage", None)
        if not usage:
            return
        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        gen_tps = (
            completion_tokens / elapsed
            if elapsed > 0 and completion_tokens > 0
            else 0
        )
        Logger.info(
            f"Token stats: prompt={prompt_tokens}, completion={completion_tokens}, "
            f"speed={gen_tps:.1f} tok/s, elapsed={elapsed:.1f}s"
        )

    async def chat(self, model, messages, gen_conf):
        t0 = time.monotonic()
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            **self._build_params(gen_conf),
        )
        self._log_token_stats(resp, time.monotonic() - t0)
        if resp.choices:
            return resp.choices[0].message.content
        return None

    async def vision_chat(self, model, prompt, image_b64, gen_conf):
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ]
        t0 = time.monotonic()
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            **self._build_params(gen_conf),
        )
        self._log_token_stats(resp, time.monotonic() - t0)
        if resp.choices:
            return resp.choices[0].message.content
        return None


_llm_backends: dict[tuple[str, str], LLMBackend] = {}


def _get_llm_backend() -> LLMBackend:
    """按 (endpoint, api_key) 缓存 backend 实例，支持多配置场景。"""
    api_key = get_llm_api_key()
    endpoint = get_llm_endpoint()
    cache_key = (endpoint, api_key)
    if cache_key not in _llm_backends:
        if api_key:
            _llm_backends[cache_key] = OpenAIBackend(
                api_key=api_key,
                base_url=endpoint,
            )
            Logger.info(
                f"Using OpenAI-compatible backend: {endpoint}"
            )
        else:
            _llm_backends[cache_key] = OllamaBackend(host=endpoint)
            Logger.info(f"Using Ollama backend: {endpoint}")
    return _llm_backends[cache_key]


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>/<thinking> wrapper tags from LLM output."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "</thinking>" in text:
        text = text.split("</thinking>")[-1]
    return text.strip()


@time_it(prefix="llm chat")
@cache_it(
    key_generator=lambda prompt, gen_conf: (
        "llm_chat::"
        + hash64(f"{get_llm_endpoint()}::{Config.chat_model_name}".encode())
        + "::prompt_hash::"
        + hash64(f"{prompt}_{json.dumps(gen_conf, default=str)}".encode())
    )
)
async def llm_chat(prompt: str, gen_conf: dict[str, Any]) -> str | None:
    messages = [{"role": "user", "content": prompt}]
    try:
        ans = await _get_llm_backend().chat(
            model=Config.chat_model_name,
            messages=messages,
            gen_conf=gen_conf,
        )
    except Exception as e:
        Logger.error(f"LLM chat exception: {e}")
        return None
    if not ans:
        return None
    return _strip_thinking_tags(ans)


@time_it(prefix="image chat")
@cache_it(
    key_generator=lambda prompt, image_content, gen_conf: (
        "image_chat::"
        + hash64(f"{get_llm_endpoint()}::{Config.vision_model_name}".encode())
        + "::prompt_hash::"
        + hash64(
            (prompt + image_content + json.dumps(gen_conf, default=str)).encode(
                "utf-8", errors="ignore"
            )
        )
    )
)
async def image_chat(
    prompt: str,
    image_content: str,
    gen_conf: dict[str, Any],
) -> str | None:
    try:
        return await _get_llm_backend().vision_chat(
            model=Config.vision_model_name,
            prompt=prompt,
            image_b64=image_content,
            gen_conf=gen_conf,
        )
    except Exception as e:
        Logger.error(f"Vision model chat exception: {e}")
        return None

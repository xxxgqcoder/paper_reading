import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any

from .config import Config, get_llm_api_key, get_llm_endpoint
from .log import Logger
from .utils import cache_it, hash64, time_it

# Ollama 本地推理并发量低，API 服务可并发更多
_CONCURRENCY_OLLAMA = 1
_CONCURRENCY_API = 10

# 兼容将本地 / Ollama 风格模型名映射为 OpenRouter 官方模型 ID
_OPENROUTER_MODEL_ALIASES = {
    "qwen3.5-flash": "qwen/qwen3.5-flash-02-23",
    "qwen2.5vl:7b": "qwen/qwen-2.5-vl-7b-instruct",
    "qwen2.5-vl:7b": "qwen/qwen-2.5-vl-7b-instruct",
}
_LOGGED_MODEL_RESOLUTIONS: set[tuple[str, str]] = set()
_LOGGED_MODEL_WARNINGS: set[tuple[str, str]] = set()


def _is_openrouter_endpoint(endpoint: str) -> bool:
    return "openrouter.ai" in endpoint


def resolve_model_name(model: str, endpoint: str | None = None) -> str:
    endpoint = endpoint or get_llm_endpoint()
    if not _is_openrouter_endpoint(endpoint):
        return model

    resolved = _OPENROUTER_MODEL_ALIASES.get(model, model)
    log_key = (endpoint, model)

    if resolved != model and log_key not in _LOGGED_MODEL_RESOLUTIONS:
        Logger.info(f"Resolved model alias for OpenRouter: {model} -> {resolved}")
        _LOGGED_MODEL_RESOLUTIONS.add(log_key)
    elif (
        resolved == model
        and "/" not in model
        and log_key not in _LOGGED_MODEL_WARNINGS
    ):
        Logger.warning(
            f"Model name '{model}' may be invalid for OpenRouter. "
            "Prefer provider-prefixed IDs such as 'qwen/qwen3-32b'."
        )
        _LOGGED_MODEL_WARNINGS.add(log_key)

    return resolved


def _get_chat_model_name(model: str | None = None) -> str:
    return resolve_model_name(model or Config.chat_model_name)


def _get_vision_model_name(model: str | None = None) -> str:
    return resolve_model_name(model or Config.vision_model_name)


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
_llm_semaphores: dict[tuple[str, str], asyncio.Semaphore] = {}


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
            _llm_semaphores[cache_key] = asyncio.Semaphore(
                _CONCURRENCY_API
            )
            Logger.info(
                f"Using OpenAI-compatible backend: {endpoint},"
                f" max_concurrency={_CONCURRENCY_API}"
            )
        else:
            _llm_backends[cache_key] = OllamaBackend(host=endpoint)
            _llm_semaphores[cache_key] = asyncio.Semaphore(
                _CONCURRENCY_OLLAMA
            )
            Logger.info(
                f"Using Ollama backend: {endpoint},"
                f" max_concurrency={_CONCURRENCY_OLLAMA}"
            )
    return _llm_backends[cache_key]


def _get_llm_semaphore() -> asyncio.Semaphore:
    """获取当前 backend 对应的并发信号量。"""
    cache_key = (get_llm_endpoint(), get_llm_api_key())
    return _llm_semaphores[cache_key]


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>/<thinking> wrapper tags from LLM output."""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "</thinking>" in text:
        text = text.split("</thinking>")[-1]
    return text.strip()


@time_it(prefix="llm chat")
@cache_it(
    key_generator=lambda prompt, gen_conf, model=None: (
        "llm_chat::"
        + hash64(f"{get_llm_endpoint()}::{_get_chat_model_name(model)}".encode())
        + "::prompt_hash::"
        + hash64(f"{prompt}_{json.dumps(gen_conf, default=str)}".encode())
    )
)
async def llm_chat(
    prompt: str,
    gen_conf: dict[str, Any],
    model: str | None = None,
) -> str | None:
    backend = _get_llm_backend()
    semaphore = _get_llm_semaphore()
    messages = [{"role": "user", "content": prompt}]
    chat_model_name = _get_chat_model_name(model)
    async with semaphore:
        try:
            ans = await backend.chat(
                model=chat_model_name,
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
    key_generator=lambda prompt, image_content, gen_conf, model=None: (
        "image_chat::"
        + hash64(f"{get_llm_endpoint()}::{_get_vision_model_name(model)}".encode())
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
    model: str | None = None,
) -> str | None:
    backend = _get_llm_backend()
    semaphore = _get_llm_semaphore()
    vision_model_name = _get_vision_model_name(model)
    async with semaphore:
        try:
            return await backend.vision_chat(
                model=vision_model_name,
                prompt=prompt,
                image_b64=image_content,
                gen_conf=gen_conf,
            )
        except Exception as e:
            Logger.error(f"Vision model chat exception: {e}")
            return None

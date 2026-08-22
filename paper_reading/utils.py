import base64
import functools
import inspect
import os
import time
from collections.abc import Callable
from typing import Any, TypeVar

import xxhash
from diskcache import Cache

from .log import Logger

line_breaker = "\n\n"

T = TypeVar("T")


def time_it(prefix: str = "") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _log_elapsed(func_name: str, begin: int) -> None:
        elapse = (time.time_ns() - begin) // 1000000
        Logger.info(
            f"{func_name} took {elapse // 60000}min"
            f" {(elapse % 60000) // 1000}sec"
            f" {elapse % 60000 % 1000}ms to finish"
        )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name = f"{prefix} {func.__name__}" if prefix else func.__name__

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*kargs, **kwargs):
                begin = time.time_ns()
                ret = await func(*kargs, **kwargs)
                _log_elapsed(func_name, begin)
                return ret
            return async_wrapper

        @functools.wraps(func)
        def wrapper(*kargs, **kwargs):
            begin = time.time_ns()
            ret = func(*kargs, **kwargs)
            _log_elapsed(func_name, begin)
            return ret
        return wrapper

    return decorator


def hash64(content: bytes) -> str:
    return xxhash.xxh64(content).hexdigest()


def safe_encode(text: str) -> str:
    try:
        return text.encode(encoding="utf-8", errors="ignore").decode(
            encoding="utf-8", errors="ignore"
        )
    except Exception as e:
        Logger.error(f"safe encode error: {e}")
        return ""


def load_base64_image(p: str) -> str:
    """Load image as base64 encoded string."""
    with open(p, "rb") as f:
        image_bytes = f.read()
        base64_string = base64.b64encode(image_bytes).decode("utf-8")
    return base64_string


def relative_md_image_path(sys_image_folder: str, img_name: str) -> str:
    return (
        f"![[{os.path.join(os.path.basename(sys_image_folder), img_name)}]]"
        + line_breaker
    )


def cache_it(
    key_generator: Callable[..., str],
    ttl_seconds=60 * 60 * 24 * 100,
    cache_data_dir: str = "~/.cache/paper-reading",
) -> Callable[..., Callable[..., T]]:
    """
    File-based cache decorator using diskcache.

    This decorator saves the results of a function call to a disk-based cache.
    On subsequent calls with the same arguments, it returns the cached result
    if it hasn't expired.

    Cache 对象在首次调用时 lazy 初始化，避免装饰时依赖外部配置。
    """
    _cache: Cache | None = None

    def _get_cache() -> Cache:
        nonlocal _cache
        if _cache is None:
            path = os.path.abspath(os.path.expanduser(cache_data_dir))
            _cache = Cache(path)
        return _cache

    def _try_get(cache_key: str):
        c = _get_cache()
        if cache_key in c:
            Logger.info(f"Loaded cache from {cache_key}")
            return True, c[cache_key]
        return False, None

    def _try_set(cache_key: str, result, ttl: int) -> None:
        if result:
            c = _get_cache()
            c.set(cache_key, result, expire=ttl)
            Logger.info(f"Saved cache to {cache_key}")

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                cache_key = key_generator(*args, **kwargs)
                hit, cached_result = _try_get(cache_key)
                if hit:
                    return cached_result  # type: ignore
                result = await func(*args, **kwargs)
                _try_set(cache_key, result, ttl_seconds)
                return result
            return async_wrapper  # type: ignore

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = key_generator(*args, **kwargs)
            hit, cached_result = _try_get(cache_key)
            if hit:
                return cached_result  # type: ignore
            result = func(*args, **kwargs)
            _try_set(cache_key, result, ttl_seconds)
            return result

        return wrapper

    return decorator

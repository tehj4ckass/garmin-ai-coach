import asyncio
import logging
import random
import re
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import Any

import anthropic
from anthropic._exceptions import DeadlineExceededError, OverloadedError, ServiceUnavailableError
from langgraph.errors import GraphInterrupt

logger = logging.getLogger(__name__)

try:
    import openai
except Exception:  # pragma: no cover
    openai = None  # type: ignore[assignment]


class RetryableError(Exception):
    pass


class APIOverloadError(RetryableError):
    pass


_RETRY_AFTER_RE = re.compile(r"try again in (?P<seconds>\d+(?:\.\d+)?)s", re.IGNORECASE)


def _suggested_retry_delay_seconds(exc: Exception) -> float | None:
    """
    Best-effort extraction of a provider-suggested retry delay.

    OpenAI rate limit errors often include: "Please try again in Xs."
    """
    try:
        msg = str(exc)
    except Exception:  # pragma: no cover
        return None
    match = _RETRY_AFTER_RE.search(msg)
    if not match:
        return None
    try:
        return float(match.group("seconds"))
    except Exception:  # pragma: no cover
        return None


class RetryConfig:

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: set[type[Exception]] | None = None,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        default_retryable_exceptions: set[type[Exception]] = {
            anthropic.RateLimitError,  # 429 - rate limits
            anthropic.ConflictError,  # 409 - conflicts
            anthropic.InternalServerError,  # 5xx - server errors
            ServiceUnavailableError,  # 503 - service unavailable
            DeadlineExceededError,  # 504 - gateway timeout
            OverloadedError,  # 529 - overloaded
            anthropic.APIConnectionError,  # Network/connection issues
            anthropic.APITimeoutError,  # Timeouts
            APIOverloadError,  # Custom local exception
        }
        if openai is not None:
            # OpenAI SDK exceptions (used by langchain_openai)
            default_retryable_exceptions.update(
                {
                    openai.RateLimitError,  # 429 - rate limits
                    openai.APIConnectionError,  # Network/connection issues
                    openai.APITimeoutError,  # Timeouts
                    openai.InternalServerError,  # 5xx
                }
            )
        self.retryable_exceptions = retryable_exceptions or default_retryable_exceptions

    def calculate_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (self.exponential_base**attempt), self.max_delay)
        if self.jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        return max(delay, 0.1)


async def retry_with_backoff(
    func: Callable[[], Awaitable[Any]],
    config: RetryConfig | None = None,
    context: str = "operation",
) -> Any:

    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            logger.debug(
                "Attempting %s (attempt %s/%s)",
                context,
                attempt + 1,
                config.max_retries + 1,
            )
            return await func()

        except GraphInterrupt:
            raise

        except Exception as e:
            last_exception = e

            is_retryable = any(isinstance(e, exc_type) for exc_type in config.retryable_exceptions)

            if is_retryable:
                logger.warning("%s failed with retryable %s: %s", context, type(e).__name__, e)
            else:
                logger.error("%s failed with non-retryable error: %s", context, e)
                break

            if attempt < config.max_retries:
                delay = config.calculate_delay(attempt)
                suggested = _suggested_retry_delay_seconds(e)
                if suggested is not None:
                    # Respect provider hints to avoid wasting retries.
                    delay = max(delay, min(suggested, config.max_delay))
                logger.info(
                    "%s failed (attempt %s), retrying in %.1fs: %s",
                    context,
                    attempt + 1,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)
            else:
                logger.error("%s failed after %s attempts: %s", context, config.max_retries + 1, e)

    if last_exception is None:
        raise RuntimeError(f"{context} failed without capturing an exception")
    raise last_exception


def with_retry(config: RetryConfig | None = None, context: str | None = None):
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_context = context or func.__name__

            async def call_func():
                return await func(*args, **kwargs)

            return await retry_with_backoff(call_func, config, func_context)

        return wrapper

    return decorator


DEFAULT_CONFIG = RetryConfig(max_retries=3, base_delay=1.0, max_delay=60.0)

AI_ANALYSIS_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=120.0,
    exponential_base=2.5,
)

QUICK_RETRY_CONFIG = RetryConfig(max_retries=2, base_delay=0.5, max_delay=10.0)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResilientLLMClient - OpenAI-compatible LLM client with automatic failover.

Supports OpenAI-compatible endpoints (Hunyuan, Qwen, etc.).
Falls back to alternative provider on failure.

Usage:
    from resilient_llm_client import ResilientLLMClient

    client = ResilientLLMClient()
    resp, metrics = client.chat(messages=[...])
    content = resp["choices"][0]["message"]["content"]
"""

import logging
import time
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

LOGGER = logging.getLogger("resilient_llm_client")


@dataclass
class RequestMetrics:
    """Metrics for a single LLM request."""
    success: bool
    latency_s: float
    attempt: int
    status_code: Optional[int]
    error: Optional[str]
    provider: str = ""


# ─────────────────── Provider Configuration ───────────────────

# Hunyuan API (OpenAI-compatible endpoint)
HUNYUAN_CONFIG = {
    "api_key": os.environ.get("HUNYUAN_API_KEY", "dMaIDnRH4iT7Tc0u8Ua8nBiv2yhNanl9"),
    "base_url": "http://hunyuanapi.woa.com",
    "model": "hunyuan-2.0-instruct-20251111",
}

# Qwen DashScope API (OpenAI-compatible endpoint)
QWEN_CONFIG = {
    "api_key": os.environ.get("QWEN_API_KEY", "sk-d5a16bb38a7646039f5715973761dd3f"),
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-plus-latest",
}


class OpenAIClientWrapper:
    """Wrapper around openai.OpenAI that matches HunyuanApiClient interface."""

    def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.endpoint = base_url

    def chat(self, messages: List[Dict], stream: bool = False, **kwargs):
        """OpenAI-compatible chat interface.

        Returns:
            Tuple of (response_dict, metrics_object)
        """
        start = time.monotonic()
        try:
            if stream:
                # Streaming mode: collect chunks
                collected = []
                for chunk in self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                ):
                    collected.append(chunk)
                # Combine chunks into full response
                content = "".join(
                    c.choices[0].delta.content or ""
                    for c in collected
                    if c.choices and c.choices[0].delta.content
                )
                resp = {
                    "choices": [{"message": {"content": content, "role": "assistant"}}],
                }
            else:
                # Non-streaming mode
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=False,
                )
                resp = {
                    "choices": [
                        {
                            "message": {
                                "content": completion.choices[0].message.content,
                                "role": completion.choices[0].message.role,
                            }
                        }
                    ],
                }

            latency = time.monotonic() - start
            metrics = type("Metrics", (), {"status_code": 200, "latency": latency})()
            return resp, metrics

        except Exception as e:
            latency = time.monotonic() - start
            status_code = getattr(e, "status_code", None)
            raise type(e)(f"OpenAI API error (status={status_code}): {e}") from e


def _create_hunyuan_client(timeout: int = 120, max_retries: int = 3, **extra_kwargs):
    """Create Hunyuan client (OpenAI-compatible)."""
    config = HUNYUAN_CONFIG
    client = OpenAIClientWrapper(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
    )
    return client


def _create_qwen_client(timeout: int = 120, max_retries: int = 3, **extra_kwargs):
    """Create Qwen client (OpenAI-compatible)."""
    config = QWEN_CONFIG
    client = OpenAIClientWrapper(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
    )
    return client


class ResilientLLMClient:
    """Dual-provider LLM client with automatic failover.

    Switches between Hunyuan and Qwen on failure.
    """

    def __init__(
        self,
        max_provider_switches: int = 6,
        switch_sleep: float = 1.0,
        timeout: int = 120,
        max_retries_per_provider: int = 3,
        primary: str = "hunyuan",
        hunyuan_kwargs: Optional[Dict[str, Any]] = None,
        qwen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.max_provider_switches = max_provider_switches
        self.switch_sleep = switch_sleep
        self.timeout = timeout
        self.max_retries_per_provider = max_retries_per_provider

        # Lazy client creation
        self._hunyuan = None
        self._qwen = None
        self._hunyuan_kwargs = hunyuan_kwargs or {}
        self._qwen_kwargs = qwen_kwargs or {}

        # Stats
        self.total_calls = 0
        self.total_switches = 0
        self.provider_stats = {
            "hunyuan": {"ok": 0, "fail": 0},
            "qwen": {"ok": 0, "fail": 0},
        }

        # Provider order
        if primary == "qwen":
            self._provider_order = ["qwen", "hunyuan"]
        else:
            self._provider_order = ["hunyuan", "qwen"]

        # Compat: expose .model attribute
        self.model = HUNYUAN_CONFIG["model"] if primary != "qwen" else QWEN_CONFIG["model"]

    def _get_client(self, name: str):
        """Get or lazily create client for provider."""
        if name == "hunyuan":
            if self._hunyuan is None:
                self._hunyuan = _create_hunyuan_client(
                    timeout=self.timeout,
                    max_retries=self.max_retries_per_provider,
                    **self._hunyuan_kwargs,
                )
                LOGGER.info(
                    "ResilientLLM: Hunyuan client created (endpoint=%s, model=%s)",
                    self._hunyuan.endpoint,
                    self._hunyuan.model,
                )
            return self._hunyuan
        else:
            if self._qwen is None:
                self._qwen = _create_qwen_client(
                    timeout=self.timeout,
                    max_retries=self.max_retries_per_provider,
                    **self._qwen_kwargs,
                )
                LOGGER.info(
                    "ResilientLLM: Qwen client created (endpoint=%s, model=%s)",
                    self._qwen.endpoint,
                    self._qwen.model,
                )
            return self._qwen

    def chat(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        request_overrides: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> Tuple[Dict[str, Any], RequestMetrics]:
        """Send chat request with automatic provider failover.

        Returns:
            Tuple of (response_dict, RequestMetrics)
        """
        self.total_calls += 1
        start_ts = time.monotonic()
        last_error: Optional[Exception] = None
        last_provider = ""

        for switch_round in range(self.max_provider_switches):
            provider_name = self._provider_order[switch_round % len(self._provider_order)]

            # Sleep before switching (not on first round)
            if switch_round > 0:
                LOGGER.info(
                    "ResilientLLM: switching to %s (round %d/%d), sleep %.1fs",
                    provider_name,
                    switch_round + 1,
                    self.max_provider_switches,
                    self.switch_sleep,
                )
                self.total_switches += 1
                time.sleep(self.switch_sleep)

            client = self._get_client(provider_name)
            last_provider = provider_name

            try:
                kwargs = {
                    "messages": messages,
                    "stream": stream,
                    "debug": debug,
                }
                if request_overrides:
                    kwargs["request_overrides"] = request_overrides
                resp, inner_metrics = client.chat(**kwargs)

                # Success
                latency = time.monotonic() - start_ts
                self.provider_stats[provider_name]["ok"] += 1

                metrics = RequestMetrics(
                    success=True,
                    latency_s=latency,
                    attempt=switch_round + 1,
                    status_code=(
                        inner_metrics.status_code
                        if hasattr(inner_metrics, "status_code")
                        else 200
                    ),
                    error=None,
                    provider=provider_name,
                )

                if switch_round > 0:
                    LOGGER.info(
                        "ResilientLLM: success via %s after %d switch(es), latency=%.1fs",
                        provider_name,
                        switch_round,
                        latency,
                    )

                return resp, metrics

            except Exception as e:
                last_error = e
                self.provider_stats[provider_name]["fail"] += 1

                error_str = str(e)[:200]
                status = getattr(e, "status_code", None)
                LOGGER.warning(
                    "ResilientLLM: %s failed (round %d/%d, status=%s): %s",
                    provider_name,
                    switch_round + 1,
                    self.max_provider_switches,
                    status,
                    error_str,
                )

        # All rounds failed
        latency = time.monotonic() - start_ts
        LOGGER.error(
            "ResilientLLM: ALL %d rounds failed (%.1fs total). "
            "stats: hunyuan=%s, qwen=%s. Last error: %s",
            self.max_provider_switches,
            latency,
            self.provider_stats["hunyuan"],
            self.provider_stats["qwen"],
            str(last_error)[:300],
        )

        # Re-raise last error
        raise last_error  # type: ignore

    def get_stats_summary(self) -> str:
        """Return human-readable stats summary."""
        return (
            f"ResilientLLM stats: calls={self.total_calls}, "
            f"switches={self.total_switches}, "
            f"hunyuan={self.provider_stats['hunyuan']}, "
            f"qwen={self.provider_stats['qwen']}"
        )

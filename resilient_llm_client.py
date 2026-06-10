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
import requests
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
# 
# To use real LLM API calls, override these via environment variables:
#   export OPENAI_API_KEY=sk-...
#   export OPENAI_BASE_URL=https://api.openai.com/v1
#   export HUNYUAN_API_KEY=...
#   export HUNYUAN_BASE_URL=http://hunyuanapi.woa.com
#   export QWEN_API_KEY=...
#   export QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

OPENAI_CONFIG = {
    "api_key": os.environ.get("OPENAI_API_KEY", ""),
    "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    "model": os.environ.get("OPENAI_MODEL", "gpt-4o"),
}

HUNYUAN_CONFIG = {
    "api_key": os.environ.get("HUNYUAN_API_KEY", "dMaIDnRH4iT7Tc0u8Ua8nBiv2yhNanl9"),
    "base_url": os.environ.get("HUNYUAN_BASE_URL", "http://hunyuanapi.woa.com/openapi"),
    "model": os.environ.get("HUNYUAN_MODEL", "hunyuan-2.0-instruct-20251111"),
}

QWEN_CONFIG = {
    "api_key": os.environ.get("QWEN_API_KEY", "sk-d5a16bb38a7646039f5715973761dd3f"),
    "base_url": os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "model": os.environ.get("QWEN_MODEL", "qwen-plus-latest"),
}


class RequestsClientWrapper:
    """HTTP client using `requests` - works with any OpenAI-compatible endpoint.
    
    Works with http:// (insecure) URLs for local testing.
    """

    def __init__(self, api_key: str, base_url: str, model: str, **kwargs):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint = base_url
        self._session = requests.Session()

    def chat(self, messages: List[Dict], stream: bool = False, **kwargs):
        """Call /v1/chat/completions, return (response_dict, metrics)."""
        start = time.monotonic()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # Always non-streaming
        }
        try:
            resp = self._session.post(
                url, headers=headers, json=payload, timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            latency = time.monotonic() - start
            metrics = RequestMetrics(
                success=True,
                latency_s=latency,
                attempt=1,
                status_code=resp.status_code,
                error=None,
                provider=self.model,
            )
            return data, metrics
        except requests.exceptions.RequestException as e:
            latency = time.monotonic() - start
            status_code = None
            if hasattr(e, "response") and e.response is not None:
                status_code = e.response.status_code
            # Re-raise with descriptive message
            raise RuntimeError(
                f"HTTP error (status={status_code}): {e}"
            ) from e


def _create_hunyuan_client(timeout: int = 120, max_retries: int = 3, **extra_kwargs):
    config = HUNYUAN_CONFIG
    return RequestsClientWrapper(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
    )


def _create_qwen_client(timeout: int = 120, max_retries: int = 3, **extra_kwargs):
    config = QWEN_CONFIG
    return RequestsClientWrapper(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
    )


def _create_openai_client(timeout: int = 120, max_retries: int = 3, **extra_kwargs):
    config = OPENAI_CONFIG
    if not config["api_key"]:
        raise ValueError(
            "OPENAI_API_KEY not set. Set via env var or pass api_key kwarg."
        )
    return RequestsClientWrapper(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
    )


class ResilientLLMClient:
    """Multi-provider LLM client with automatic failover."""

    def __init__(
        self,
        max_provider_switches: int = 6,
        switch_sleep: float = 1.0,
        timeout: int = 120,
        max_retries_per_provider: int = 3,
        primary: str = "hunyuan",
        openai_kwargs: Optional[Dict[str, Any]] = None,
        hunyuan_kwargs: Optional[Dict[str, Any]] = None,
        qwen_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.max_provider_switches = max_provider_switches
        self.switch_sleep = switch_sleep
        self.timeout = timeout
        self.max_retries_per_provider = max_retries_per_provider

        self._openai = None
        self._hunyuan = None
        self._qwen = None
        self._openai_kwargs = openai_kwargs or {}
        self._hunyuan_kwargs = hunyuan_kwargs or {}
        self._qwen_kwargs = qwen_kwargs or {}

        self.total_calls = 0
        self.total_switches = 0
        self.provider_stats = {
            "openai": {"ok": 0, "fail": 0},
            "hunyuan": {"ok": 0, "fail": 0},
            "qwen": {"ok": 0, "fail": 0},
        }

        providers = ["openai", "hunyuan", "qwen"]
        if primary in providers:
            providers.remove(primary)
            self._provider_order = [primary] + providers
        else:
            self._provider_order = [primary] + [p for p in providers if p != primary]

        if primary == "openai":
            self.model = OPENAI_CONFIG["model"]
        elif primary == "hunyuan":
            self.model = HUNYUAN_CONFIG["model"]
        else:
            self.model = QWEN_CONFIG["model"]

    def _get_client(self, name: str):
        if name == "openai":
            if self._openai is None:
                self._openai = _create_openai_client(**self._openai_kwargs)
                LOGGER.info("ResilientLLM: OpenAI client created")
            return self._openai
        elif name == "hunyuan":
            if self._hunyuan is None:
                self._hunyuan = _create_hunyuan_client(**self._hunyuan_kwargs)
                LOGGER.info("ResilientLLM: Hunyuan client created")
            return self._hunyuan
        else:
            if self._qwen is None:
                self._qwen = _create_qwen_client(**self._qwen_kwargs)
                LOGGER.info("ResilientLLM: Qwen client created")
            return self._qwen

    def chat(
        self,
        messages: List[Dict[str, Any]],
        stream: bool = False,
        request_overrides: Optional[Dict[str, Any]] = None,
        debug: bool = False,
    ) -> Tuple[Dict[str, Any], RequestMetrics]:
        self.total_calls += 1
        start_ts = time.monotonic()
        last_error = None

        for switch_round in range(self.max_provider_switches):
            provider_name = self._provider_order[switch_round % len(self._provider_order)]

            if switch_round > 0:
                LOGGER.info(
                    "ResilientLLM: switching to %s (round %d/%d)",
                    provider_name, switch_round + 1, self.max_provider_switches,
                )
                self.total_switches += 1
                time.sleep(self.switch_sleep)

            client = self._get_client(provider_name)
            try:
                resp, inner_metrics = client.chat(
                    messages=messages, stream=False,
                )
                latency = time.monotonic() - start_ts
                self.provider_stats[provider_name]["ok"] += 1
                metrics = RequestMetrics(
                    success=True,
                    latency_s=latency,
                    attempt=switch_round + 1,
                    status_code=getattr(inner_metrics, "status_code", 200),
                    error=None,
                    provider=provider_name,
                )
                if switch_round > 0:
                    LOGGER.info(
                        "ResilientLLM: success via %s after %d switch(es)",
                        provider_name, switch_round,
                    )
                return resp, metrics
            except Exception as e:
                last_error = e
                self.provider_stats[provider_name]["fail"] += 1
                LOGGER.warning(
                    "ResilientLLM: %s failed (round %d/%d): %s",
                    provider_name, switch_round + 1,
                    self.max_provider_switches, str(e)[:200],
                )

        latency = time.monotonic() - start_ts
        LOGGER.error(
            "ResilientLLM: ALL %d rounds failed. Last: %s",
            self.max_provider_switches, str(last_error)[:300],
        )
        raise RuntimeError(f"All providers failed: {last_error}") from last_error

    def get_stats_summary(self) -> str:
        return (
            f"calls={self.total_calls}, "
            f"switches={self.total_switches}, "
            f"hunyuan={self.provider_stats['hunyuan']}"
        )

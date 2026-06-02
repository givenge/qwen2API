from __future__ import annotations

from typing import Any


_FALSE_STRINGS = {"0", "false", "no", "off", "disabled", "disable", "none"}
_TRUE_STRINGS = {"1", "true", "yes", "on", "enabled", "enable", "auto"}
_REASONING_EFFORT_STRINGS = {"minimal", "low", "medium", "high"}


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _FALSE_STRINGS:
            return False
        if normalized in _TRUE_STRINGS:
            return True
    return None


def _coerce_reasoning_effort(value: Any) -> bool | None:
    coerced = _coerce_bool(value)
    if coerced is not None:
        return coerced
    if isinstance(value, str) and value.strip().lower() in _REASONING_EFFORT_STRINGS:
        return True
    return None


def _extract_thinking_object(value: Any) -> bool | None:
    coerced = _coerce_bool(value)
    if coerced is not None:
        return coerced
    if not isinstance(value, dict):
        return None

    enabled = _coerce_bool(value.get("enabled"))
    if enabled is not None:
        return enabled

    kind = _coerce_bool(value.get("type"))
    if kind is not None:
        return kind

    budget_tokens = value.get("budget_tokens")
    if isinstance(budget_tokens, int):
        return budget_tokens > 0

    effort = _coerce_reasoning_effort(value.get("effort"))
    if effort is not None:
        return effort

    return None


def extract_request_thinking_enabled(payload: dict[str, Any]) -> bool | None:
    """Normalize OpenAI/Anthropic/frontend thinking controls to thinking_enabled."""
    for key in ("thinking_enabled", "enable_thinking", "thinkingEnabled"):
        value = _coerce_bool(payload.get(key))
        if value is not None:
            return value

    thinking = _extract_thinking_object(payload.get("thinking"))
    if thinking is not None:
        return thinking

    reasoning = _extract_thinking_object(payload.get("reasoning"))
    if reasoning is not None:
        return reasoning

    return _coerce_reasoning_effort(payload.get("reasoning_effort"))

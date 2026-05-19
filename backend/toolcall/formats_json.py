from __future__ import annotations

import json
import re
from typing import Any

from .normalize import normalize_arguments, normalize_tool_name


JSON_INPUT_KEYS = ("input", "arguments", "args", "parameters")
_MARKER_RE = re.compile(r"##TOOL_CALL##\s*(.*?)\s*##END_CALL##", re.DOTALL | re.IGNORECASE)
_FENCED_JSON_RE = re.compile(r"```(?:json|tool_call)?\s*\n?([\s\S]*?)\n?```", re.IGNORECASE)


def _repair_loose_json(text: str) -> str:
    repaired = text.strip()
    if not repaired:
        return repaired
    repaired = repaired.replace('"name="', '"name": "')
    repaired = re.sub(r'"name=([^",}]+)"', r'"name": "\1"', repaired)
    repaired = re.sub(r'"name=([^",}]+)', r'"name": "\1"', repaired)
    repaired = re.sub(r'"name\s*=\s*"', '"name": "', repaired)
    repaired = re.sub(r'"(name|input|arguments|args|parameters)"\s*=\s*', r'"\1": ', repaired)
    return repaired


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    candidates: list[str] = []
    marker_match = _MARKER_RE.search(stripped)
    if marker_match:
        candidates.append(marker_match.group(1).strip())

    fence_match = _FENCED_JSON_RE.search(stripped)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    candidates.append(stripped)
    return candidates


def _extract_call(payload: object, allowed_names: set[str]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None

    # Qwen / OpenAI 官方单对象格式：{"id":"...","type":"function","function":{"name":...,"arguments":"..."}}
    # name 和 arguments 嵌在 function 子字段里，不是顶层。向下穿透一层再提取。
    nested_function = payload.get("function")
    if isinstance(nested_function, dict) and nested_function.get("name"):
        payload = nested_function

    name = payload.get("name")
    if not name:
        return None

    raw_input = payload.get("input")
    if "input" not in payload:
        for key in JSON_INPUT_KEYS[1:]:
            if key in payload:
                raw_input = payload[key]
                break
        else:
            raw_input = {}
    return {
        "name": name if isinstance(name, str) and name in allowed_names else normalize_tool_name(name, allowed_names),
        "input": normalize_arguments(raw_input),
    }


def _parse_json_payload(stripped: str) -> object | None:
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        repaired = _repair_loose_json(stripped)
        if repaired == stripped:
            return None
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None


def _payload_calls(payload: object, allowed_names: set[str]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
        calls = []
        for item in payload["tool_calls"]:
            if not isinstance(item, dict):
                continue
            function_payload = item.get("function")
            if not isinstance(function_payload, dict):
                continue
            call = _extract_call(function_payload, allowed_names)
            if call:
                calls.append(call)
        return calls

    call = _extract_call(payload, allowed_names) if isinstance(payload, dict) else None
    return [call] if call else []


def parse_json_format(text: str, allowed_names: set[str]) -> list[dict[str, Any]]:
    for candidate in _json_candidates(text):
        payload = _parse_json_payload(candidate)
        calls = _payload_calls(payload, allowed_names)
        if calls:
            return calls
    return []

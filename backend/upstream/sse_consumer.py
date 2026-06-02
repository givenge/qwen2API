import json
import logging

log = logging.getLogger("qwen2api.sse")


def _summary_content_text(value) -> str:
    if not isinstance(value, dict):
        return ""
    content = value.get("content")
    if isinstance(content, list):
        return "\n".join(str(item) for item in content if item)
    if isinstance(content, str):
        return content
    return ""


def _extract_thinking_summary(extra) -> str:
    if not isinstance(extra, dict):
        return ""
    title = _summary_content_text(extra.get("summary_title"))
    thought = _summary_content_text(extra.get("summary_thought"))
    parts = []
    if title:
        parts.append(f"### {title}")
    if thought:
        parts.append(thought)
    return "\n\n".join(parts)


def parse_sse_chunk(chunk: str) -> list[dict]:
    events = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            obj = json.loads(data)
            events.append(obj)
        except Exception:
            continue

    parsed = []
    for evt in events:
        if evt.get("choices"):
            delta = evt["choices"][0].get("delta", {})
            content = delta.get("content", "")
            reasoning_snapshot = False
            if not content and delta.get("phase") == "thinking_summary":
                summary = _extract_thinking_summary(delta.get("extra"))
                if summary:
                    content = summary
                    reasoning_snapshot = True

            # Log if content contains "Tool" and "does not exist"
            if content and "Tool" in content and "does not exist" in content:
                log.warning(f"[SSE] Detected tool interception: content={content!r} phase={delta.get('phase')} status={delta.get('status')} extra={delta.get('extra')}")

            parsed.append(
                {
                    "type": "delta",
                    "phase": delta.get("phase", "answer"),
                    "content": content,
                    "status": delta.get("status", ""),
                    "extra": delta.get("extra", {}),
                    "reasoning_snapshot": reasoning_snapshot,
                }
            )
    return parsed

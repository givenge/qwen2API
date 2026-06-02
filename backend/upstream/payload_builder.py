import os
import time
import uuid

# 默认关闭 thinking 以降低首字时延（thinking 阶段可消耗 10-30s）。
# 设置 THINKING_ENABLED=true 环境变量可重新开启。
_THINKING_ENABLED = os.getenv("THINKING_ENABLED", "false").lower() in ("true", "1", "yes")

_BASE_FEATURE_CONFIG = {
    "output_schema": "phase",
    "research_mode": "normal",
    "thinking_format": "summary",
    "auto_search": False,
    "code_interpreter": False,
    "plugins_enabled": False,
}


def build_chat_payload(chat_id: str, model: str, content: str, has_custom_tools: bool = False, files: list[dict] | None = None, thinking_enabled: bool | None = None) -> dict:
    ts = int(time.time())
    # thinking 优先级：模型/请求显式值 > 工具请求默认关闭 > 全局默认。
    # 工具请求默认关闭是为了稳定本地工具解析；但 thinking 版本模型必须能显式打开。
    if thinking_enabled is not None:
        effective_thinking = thinking_enabled
    elif has_custom_tools:
        effective_thinking = False
    else:
        effective_thinking = _THINKING_ENABLED
    feature_config = {
        **_BASE_FEATURE_CONFIG,
        "thinking_enabled": effective_thinking,
        "auto_thinking": effective_thinking,
        "thinking_mode": "Auto" if effective_thinking else "disabled",
        # Our Anthropic/OpenAI bridge relies on textual JSON/XML tool directives
        # that are parsed locally. Enabling Qwen native function_calling here causes
        # upstream interception such as `Tool Read/Bash does not exists.` for custom
        # local tools that only exist in the bridge layer.
        "function_calling": False,
        "enable_tools": False,
        "enable_function_call": False,
        "tool_choice": "none",
    }
    return {
        "stream": True,
        "version": "2.1",
        "incremental_output": True,
        "chat_id": chat_id,
        "chat_mode": "normal",
        "model": model,
        "parent_id": None,
        "messages": [
            {
                "fid": str(uuid.uuid4()),
                "parentId": None,
                "childrenIds": [str(uuid.uuid4())],
                "role": "user",
                "content": content,
                "user_action": "chat",
                "files": files or [],
                "timestamp": ts,
                "models": [model],
                "chat_type": "t2t",
                "feature_config": feature_config,
                "extra": {"meta": {"subChatType": "t2t"}},
                "sub_chat_type": "t2t",
                "parent_id": None,
            }
        ],
        "timestamp": ts,
    }

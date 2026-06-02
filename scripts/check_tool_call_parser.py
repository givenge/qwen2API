import unittest
import asyncio
from pathlib import Path
import sys
import types
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pydantic_settings  # noqa: F401
except ModuleNotFoundError:
    pydantic_settings = types.ModuleType("pydantic_settings")

    class BaseSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    pydantic_settings.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = pydantic_settings

from backend.adapter.standard_request import CLAUDE_CODE_OPENAI_PROFILE
from backend.adapter.standard_request import StandardRequest
from backend.core.config import resolve_model_config
from backend.runtime.execution import RuntimeAttemptState, RuntimeToolDirective, collect_completion_run
from backend.services import tool_parser
from backend.services.openai_stream_translator import OpenAIStreamTranslator
from backend.services.response_formatters import build_openai_completion_payload
from backend.services.standard_request_builder import build_chat_standard_request
from backend.services.thinking_control import extract_request_thinking_enabled
from backend.toolcall.parser import parse_tool_calls_detailed
from backend.upstream.payload_builder import build_chat_payload


TOOLS = [
    {"name": "Read", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}}},
    {"name": "Bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}},
]
TOOL_NAMES = {tool["name"] for tool in TOOLS}


class ToolCallParserTests(unittest.TestCase):
    def assert_first_call(self, text, name, input_data):
        detailed = parse_tool_calls_detailed(text, TOOL_NAMES)
        self.assertTrue(detailed["saw_tool_syntax"])
        self.assertEqual(detailed["calls"][0]["name"], name)
        self.assertEqual(detailed["calls"][0]["input"], input_data)

    def test_detailed_parser_accepts_marker_format(self):
        self.assert_first_call(
            '##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "/tmp/a.txt"}}\n##END_CALL##',
            "Read",
            {"file_path": "/tmp/a.txt"},
        )

    def test_detailed_parser_accepts_tool_call_codeblock(self):
        self.assert_first_call(
            '```tool_call\n{"name": "Bash", "input": {"command": "pwd"}}\n```',
            "Bash",
            {"command": "pwd"},
        )

    def test_service_parser_accepts_supported_formats(self):
        samples = [
            '##TOOL_CALL##\n{"name": "Read", "input": {"file_path": "/tmp/a.txt"}}\n##END_CALL##',
            '<tool_call>{"name": "Read", "input": {"file_path": "/tmp/a.txt"}}</tool_call>',
            '```tool_call\n{"name": "Read", "input": {"file_path": "/tmp/a.txt"}}\n```',
            '{"tool_calls": [{"function": {"name": "Read", "arguments": "{\\"file_path\\": \\"/tmp/a.txt\\"}"}}]}',
            '{"name": "Read", "input": {"file_path": "/tmp/a.txt"}}',
            'function.name: Read\nfunction.arguments: {"file_path": "/tmp/a.txt"}',
        ]
        for sample in samples:
            with self.subTest(sample=sample):
                blocks, stop_reason = tool_parser.parse_tool_calls_silent(sample, TOOLS)
                self.assertEqual(stop_reason, "tool_use")
                tool_blocks = [block for block in blocks if block.get("type") == "tool_use"]
                self.assertEqual(len(tool_blocks), 1)
                self.assertEqual(tool_blocks[0]["name"], "Read")
                self.assertEqual(tool_blocks[0]["input"], {"file_path": "/tmp/a.txt"})

    def test_service_parser_leaves_plain_text_untouched(self):
        blocks, stop_reason = tool_parser.parse_tool_calls_silent("hello there", TOOLS)
        self.assertEqual(stop_reason, "end_turn")
        self.assertEqual(blocks, [{"type": "text", "text": "hello there"}])

    def test_tool_sieve_parses_once_after_complete_marker_json(self):
        calls = []
        original_parser = tool_parser.parse_tool_calls_silent

        def counting_parser(answer, tools):
            calls.append(answer)
            return original_parser(answer, tools)

        tool_parser.parse_tool_calls_silent = counting_parser
        try:
            sieve = tool_parser.ToolSieve(["Read"])
            self.assertEqual(sieve.process_chunk("before ##TOOL_CALL##\n"), [{"type": "content", "text": "before "}])
            self.assertEqual(sieve.process_chunk('{"name": "Read", '), [])

            events = sieve.process_chunk('"input": {"file_path": "/tmp/a.txt"}}\n')
            self.assertEqual(len(calls), 1)
            self.assertEqual(events, [{"type": "tool_calls", "calls": [{"name": "Read", "input": {"file_path": "/tmp/a.txt"}}]}])
        finally:
            tool_parser.parse_tool_calls_silent = original_parser

    def test_tool_sieve_accepts_complete_marker_json_without_end_marker(self):
        sieve = tool_parser.ToolSieve(["Bash"])
        self.assertEqual(sieve.process_chunk("##TOOL_CALL##\n"), [])

        events = sieve.process_chunk('{"name": "shell_run", "input": {"command": "ls -la"}}')

        self.assertEqual(events, [{"type": "tool_calls", "calls": [{"name": "Bash", "input": {"command": "ls -la"}}]}])

    def test_tool_sieve_accepts_complete_marker_json_in_first_chunk(self):
        sieve = tool_parser.ToolSieve(["Bash"])

        events = sieve.process_chunk('##TOOL_CALL##\n{"name": "shell_run", "input": {"command": "ls -la"}}')

        self.assertEqual(events, [{"type": "tool_calls", "calls": [{"name": "Bash", "input": {"command": "ls -la"}}]}])

    def test_tool_sieve_holds_split_marker_after_long_prefix(self):
        sieve = tool_parser.ToolSieve(["Bash"])
        prefix = "x" * 30

        self.assertEqual(sieve.process_chunk(prefix + "##TOOL_CALL"), [{"type": "content", "text": prefix}])
        events = sieve.process_chunk('##\n{"name": "shell_run", "input": {"command": "ls -la"}}')

        self.assertEqual(events, [{"type": "tool_calls", "calls": [{"name": "Bash", "input": {"command": "ls -la"}}]}])

    def test_qwen_thinking_variant_resolution(self):
        cases = {
            "qwen-3.6plus-thinking": ("qwen3.6-plus", True),
            "qwen-3.6plus-nonthinking": ("qwen3.6-plus", False),
            "qwen-3.6plus-nonthiking": ("qwen3.6-plus", False),
            "qwen-3.7max-thinking": ("qwen3.7-max-preview", True),
            "qwen-3.7max-nonthinking": ("qwen3.7-max-preview", False),
        }
        for model, expected in cases.items():
            with self.subTest(model=model):
                resolution = resolve_model_config(model)
                self.assertEqual((resolution.resolved_model, resolution.thinking_enabled), expected)

    def test_explicit_thinking_survives_tool_mode(self):
        payload = build_chat_payload(
            "chat",
            "qwen3.6-plus",
            "hello",
            has_custom_tools=True,
            thinking_enabled=True,
        )
        feature_config = payload["messages"][0]["feature_config"]
        self.assertTrue(feature_config["thinking_enabled"])
        self.assertTrue(feature_config["auto_thinking"])
        self.assertEqual(feature_config["thinking_mode"], "Auto")

        default_tool_payload = build_chat_payload(
            "chat",
            "qwen3.6-plus",
            "hello",
            has_custom_tools=True,
            thinking_enabled=None,
        )
        self.assertFalse(default_tool_payload["messages"][0]["feature_config"]["thinking_enabled"])

    def test_protocol_thinking_formats_normalize_to_thinking_enabled(self):
        cases = [
            ({"thinking_enabled": True}, True),
            ({"thinking_enabled": False}, False),
            ({"thinking": {"type": "enabled", "budget_tokens": 1024}}, True),
            ({"thinking": {"type": "disabled"}}, False),
            ({"thinking": {"enabled": True}}, True),
            ({"reasoning": {"effort": "high"}}, True),
            ({"reasoning_effort": "none"}, False),
        ]
        for payload, expected in cases:
            with self.subTest(payload=payload):
                self.assertEqual(extract_request_thinking_enabled(payload), expected)

    def test_request_thinking_enabled_uses_model_suffix_priority(self):
        request = build_chat_standard_request(
            {
                "model": "qwen-3.6plus-nonthinking",
                "thinking_enabled": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
            default_model="gpt-3.5-turbo",
            surface="openai",
        )

        self.assertFalse(request.thinking_enabled)
        self.assertFalse(request.model_thinking_enabled)

    def test_openai_stream_translator_emits_tool_call_delta(self):
        def build_directive(_answer_text):
            return RuntimeToolDirective(
                tool_blocks=[
                    {
                        "type": "tool_use",
                        "id": "toolu_test",
                        "name": "Read",
                        "input": {"file_path": "/tmp/a.txt"},
                    }
                ],
                stop_reason="tool_use",
            )

        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl-test",
            created=1,
            model_name="qwen-3.6plus-nonthinking",
            client_profile=CLAUDE_CODE_OPENAI_PROFILE,
            build_final_directive=build_directive,
            allowed_tool_names=["Read"],
        )
        translator.on_delta(
            {"phase": "answer"},
            '##TOOL_CALL##\n{"name":"Read","input":{"file_path":"/tmp/a.txt"}}\n##END_CALL##',
            None,
        )

        output = "".join(translator.finalize("tool_calls"))
        self.assertIn('"tool_calls"', output)
        self.assertIn('"finish_reason": "tool_calls"', output)
        self.assertNotIn("##TOOL_CALL##", output)

    def test_openai_stream_translator_uses_runtime_final_directive(self):
        translator = OpenAIStreamTranslator(
            completion_id="chatcmpl-test",
            created=1,
            model_name="qwen-3.6plus-thinking",
            client_profile=CLAUDE_CODE_OPENAI_PROFILE,
            build_final_directive=lambda _answer_text: RuntimeToolDirective(),
            allowed_tool_names=["Read"],
        )
        runtime_directive = RuntimeToolDirective(
            tool_blocks=[
                {
                    "type": "tool_use",
                    "id": "toolu_runtime",
                    "name": "Read",
                    "input": {"file_path": "/tmp/from-runtime.txt"},
                }
            ],
            stop_reason="tool_use",
        )

        output = "".join(translator.finalize("tool_calls", directive=runtime_directive))
        self.assertIn('"tool_calls"', output)
        self.assertIn("toolu_runtime", output)
        self.assertIn("/tmp/from-runtime.txt", output)
        self.assertIn('"finish_reason": "tool_calls"', output)

    def test_openai_non_stream_response_preserves_reasoning_content(self):
        request = StandardRequest(
            prompt="hello",
            response_model="qwen-3.6plus-thinking",
            resolved_model="qwen3.6-plus",
            surface="openai",
        )
        execution = SimpleNamespace(
            state=RuntimeAttemptState(
                answer_text="final answer",
                reasoning_text="reasoning summary",
            )
        )

        payload = build_openai_completion_payload(
            completion_id="chatcmpl-test",
            created=1,
            model_name=request.response_model,
            prompt=request.prompt,
            execution=execution,
            standard_request=request,
        )

        message = payload["choices"][0]["message"]
        self.assertEqual(message["content"], "final answer")
        self.assertEqual(message["reasoning_content"], "reasoning summary")

    def test_answer_phase_markdown_thinking_is_split_into_reasoning(self):
        async def run_case():
            class FakeClient:
                async def chat_stream_events_with_retry(self, *args, **kwargs):
                    yield {"type": "meta", "chat_id": "chat", "acc": None}
                    for content in (
                        "## 思考过程\n\n",
                        "先比较小数位。",
                        "\n\n## 最终答案\n\n",
                        "9.9",
                    ):
                        yield {
                            "type": "event",
                            "event": {
                                "type": "delta",
                                "phase": "answer",
                                "content": content,
                            },
                        }

            request = StandardRequest(
                prompt="hello",
                response_model="qwen-3.6plus-thinking",
                resolved_model="qwen3.6-plus",
                surface="openai",
                thinking_enabled=True,
            )
            streamed: list[tuple[str, str]] = []

            async def on_delta(evt, text_chunk, _tool_calls):
                if text_chunk:
                    streamed.append((evt.get("phase"), text_chunk))

            result = await collect_completion_run(
                FakeClient(),
                request,
                request.prompt,
                capture_events=False,
                on_delta=on_delta,
            )

            self.assertIn("先比较小数位", result.state.reasoning_text)
            self.assertEqual(result.state.answer_text, "9.9")
            self.assertEqual(streamed[0][0], "thinking_summary")
            self.assertIn("先比较小数位", streamed[0][1])
            self.assertEqual(streamed[-1], ("answer", "9.9"))

        asyncio.run(run_case())

    def test_reasoning_tool_call_marker_is_not_streamed_to_client(self):
        async def run_case():
            class FakeClient:
                async def chat_stream_events_with_retry(self, *args, **kwargs):
                    yield {"type": "meta", "chat_id": "chat", "acc": None}
                    yield {
                        "type": "event",
                        "event": {
                            "type": "delta",
                            "phase": "think",
                            "content": "checking\n##TOOL_CALL##\n",
                        },
                    }
                    yield {
                        "type": "event",
                        "event": {
                            "type": "delta",
                            "phase": "think",
                            "content": '{"name":"Read","input":{"file_path":"/tmp/a.txt"}}',
                        },
                    }

            request = StandardRequest(
                prompt="hello",
                response_model="qwen-3.6plus-thinking",
                resolved_model="qwen3.6-plus",
                surface="openai",
                tools=TOOLS,
                tool_names=["Read", "Bash"],
                tool_enabled=True,
                thinking_enabled=True,
            )
            streamed: list[str] = []

            async def on_delta(_evt, text_chunk, _tool_calls):
                if text_chunk:
                    streamed.append(text_chunk)

            result = await collect_completion_run(
                FakeClient(),
                request,
                request.prompt,
                capture_events=False,
                on_delta=on_delta,
            )

            self.assertEqual([call["name"] for call in result.state.tool_calls], ["Read"])
            self.assertEqual(result.state.tool_calls[0]["input"], {"file_path": "/tmp/a.txt"})
            self.assertEqual("".join(streamed), "checking\n")
            self.assertNotIn("##TOOL_CALL##", "".join(streamed))

        asyncio.run(run_case())

if __name__ == "__main__":
    unittest.main()

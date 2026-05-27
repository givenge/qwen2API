import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services import tool_parser
from backend.toolcall.parser import parse_tool_calls_detailed


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


if __name__ == "__main__":
    unittest.main()

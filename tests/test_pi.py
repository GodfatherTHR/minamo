import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pi_agent  # noqa: E402
from config import Config  # noqa: E402
from store import Store  # noqa: E402
from tools import Workspace  # noqa: E402


class FakeTelegram:
    def __init__(self):
        self.sent: list[str] = []
        self.edits: list[str] = []
        self.markups: list[dict | None] = []
        self._next = 700

    def send_message(self, chat_id, text, reply_markup=None, reply_to=None):
        self.sent.append(text)
        self.markups.append(reply_markup)
        self._next += 1
        return {"message_id": self._next}

    def edit_message(self, chat_id, message_id, text, reply_markup=None):
        self.edits.append(text)

    def send_typing(self, chat_id):
        pass

    def answer_callback(self, callback_id, text=""):
        pass


FAKE_PI = r'''#!/usr/bin/env python3
import json, sys
events = [
    {"type": "session", "version": 3, "id": "sess-1", "cwd": "/tmp"},
    {"type": "agent_start"},
    {"type": "message_update", "assistantMessageEvent": {"type": "thinking_delta", "delta": "hmm"}},
    {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "Working on it"}},
    {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "bash", "args": {"command": "ls"}},
    {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "bash",
     "result": {"content": [{"type": "text", "text": "a.txt"}]}, "isError": False, "durationMs": 120},
    {"type": "message_end", "message": {"role": "assistant",
     "content": [{"type": "text", "text": "Done: a.txt exists."}]}},
    {"type": "agent_settled", "aborted": False},
]
for event in events:
    print(json.dumps(event))
'''


class PiModelConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models_file = Path(self.tmp.name) / "models.json"
        self.models_file.write_text(json.dumps({"providers": {"ollama": {"models": [{"id": "qwen3:8b"}]}}}))
        self.old = os.environ.get("PI_MODELS_FILE")
        os.environ["PI_MODELS_FILE"] = str(self.models_file)

    def tearDown(self):
        if self.old is None:
            os.environ.pop("PI_MODELS_FILE", None)
        else:
            os.environ["PI_MODELS_FILE"] = self.old
        self.tmp.cleanup()

    def test_lists_configured_models(self):
        self.assertEqual(pi_agent.pi_model_ids(), ["qwen3:8b"])

    def test_registers_new_model(self):
        self.assertTrue(pi_agent.ensure_pi_model("beam:cloud"))
        self.assertIn("beam:cloud", pi_agent.pi_model_ids())
        self.assertFalse(pi_agent.ensure_pi_model("beam:cloud"))  # idempotent

    def test_missing_file_is_not_fatal(self):
        os.environ["PI_MODELS_FILE"] = str(Path(self.tmp.name) / "nope.json")
        self.assertEqual(pi_agent.pi_model_ids(), [])
        self.assertFalse(pi_agent.ensure_pi_model("x"))


class PiRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fake_pi = self.root / "fake-pi"
        self.fake_pi.write_text(FAKE_PI)
        self.fake_pi.chmod(0o755)
        self.cfg = Config()
        self.cfg.pi_bin = str(self.fake_pi)
        self.cfg.pi_model = "beam:cloud"
        self.cfg.pi_approval_timeout = 5
        self.cfg.data_dir = self.root
        self.cfg.edit_interval = 0
        self.store = Store(self.root / "bot.db")
        self.tg = FakeTelegram()
        self.runner = pi_agent.PiRunner(
            self.cfg, self.store, self.tg, Workspace(self.root)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_streams_text_and_tool_calls(self):
        self.runner.start(1, "list files", self.root)
        transcript = "\n".join(self.tg.sent + self.tg.edits)
        self.assertIn("Done: a.txt exists.", transcript)
        self.assertIn("bash", transcript)
        self.assertIn("a.txt", transcript)
        self.assertTrue(any(text.startswith("⏱") for text in self.tg.sent))

    def test_run_state_is_cleared_afterwards(self):
        self.runner.start(2, "do a thing", self.root)
        self.assertIsNone(self.store.get_state("pi_running:2"))
        self.assertFalse(self.runner.running(2))

    def test_missing_binary_reports_error(self):
        self.cfg.pi_bin = str(self.root / "does-not-exist")
        self.runner.start(3, "task", self.root)
        self.assertTrue(any("could not start pi" in text for text in self.tg.sent))


class ApprovalBridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = Config()
        self.cfg.pi_approval_timeout = 10
        self.cfg.agent_auto = False
        self.cfg.data_dir = self.root
        self.store = Store(self.root / "bot.db")
        self.tg = FakeTelegram()
        self.runner = pi_agent.PiRunner(
            self.cfg, self.store, self.tg, Workspace(self.root)
        )
        self.runner.owner_id = 42
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.server = pi_agent.ApprovalServer(self.runner, self.port)
        self.assertTrue(self.server.start())

    def tearDown(self):
        self.server.stop()
        self.tmp.cleanup()

    def _post(self, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/approve",
            data=data,
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.loads(response.read().decode())

    def _wait_for_approval(self, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            pending = self.store.pending_approvals(42)
            if pending:
                return pending[0]
            time.sleep(0.05)
        self.fail("no approval was requested")

    def test_approving_allows_the_tool(self):
        result = {}

        def call():
            result.update(self._post({"chat_id": 42, "tool": "bash", "args": {"command": "ls"}}))

        thread = threading.Thread(target=call)
        thread.start()
        approval = self._wait_for_approval()
        self.assertTrue(any("Approve" in (m or {}).get("inline_keyboard", [[{}]])[0][0]["text"]
                            for m in self.tg.markups if m))
        self.runner.decide(approval["id"], True)
        thread.join(timeout=10)
        self.assertEqual(result.get("allow"), True)

    def test_denying_blocks_the_tool(self):
        result = {}

        def call():
            result.update(self._post({"chat_id": 42, "tool": "bash", "args": {"command": "rm x"}}))

        thread = threading.Thread(target=call)
        thread.start()
        approval = self._wait_for_approval()
        self.runner.decide(approval["id"], False)
        thread.join(timeout=10)
        self.assertEqual(result.get("allow"), False)

    def test_read_only_tools_skip_approval(self):
        result = self._post({"chat_id": 42, "tool": "read", "args": {"path": "x"}})
        self.assertTrue(result["allow"])
        self.assertEqual(self.store.pending_approvals(42), [])

    def test_auto_mode_skips_approval(self):
        self.store.set_state("auto:42", "1")
        result = self._post({"chat_id": 42, "tool": "bash", "args": {"command": "ls"}})
        self.assertTrue(result["allow"])
        self.assertEqual(self.store.pending_approvals(42), [])

    def test_unknown_chat_is_refused(self):
        result = self._post({"chat_id": 999, "tool": "bash", "args": {"command": "ls"}})
        self.assertFalse(result["allow"])


@unittest.skipUnless(os.environ.get("MINAMO_LIVE_PI"), "set MINAMO_LIVE_PI=1 to hit the real model")
class LivePiTests(unittest.TestCase):
    def test_pi_creates_a_file(self):
        import shutil

        if not shutil.which("pi"):
            self.skipTest("pi not installed")
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        cfg = Config()
        cfg.data_dir = root
        cfg.workspace_root = root
        cfg.agent_auto = True
        cfg.pi_timeout = 240
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cfg.approval_port = sock.getsockname()[1]
        store = Store(root / "bot.db")
        tg = FakeTelegram()
        runner = pi_agent.PiRunner(cfg, store, tg, Workspace(root))
        runner.owner_id = 7
        server = pi_agent.ApprovalServer(runner, cfg.approval_port)
        self.assertTrue(server.start(), "approval bridge did not start")
        try:
            runner.start(7, "Create a file named proof.txt containing exactly: pi works", root)
            self.assertTrue((root / "proof.txt").exists(), "\n".join(tg.sent + tg.edits))
            self.assertIn("pi works", (root / "proof.txt").read_text())
        finally:
            server.stop()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)

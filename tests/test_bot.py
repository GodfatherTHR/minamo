import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bot  # noqa: E402


class SplitReasoningTests(unittest.TestCase):
    def test_plain_answer(self):
        thinking, answer = bot.split_reasoning("hello there")
        self.assertEqual(thinking, "")
        self.assertEqual(answer, "hello there")

    def test_closed_think_block(self):
        raw = "<think>2+2 is 4</think>\nThe answer is 4."
        thinking, answer = bot.split_reasoning(raw)
        self.assertEqual(thinking, "2+2 is 4")
        self.assertEqual(answer.strip(), "The answer is 4.")

    def test_unclosed_think_block_hides_reasoning(self):
        thinking, answer = bot.split_reasoning("<think>still thinking about it")
        self.assertEqual(thinking, "still thinking about it")
        self.assertEqual(answer, "")

    def test_partial_tag_is_held_back(self):
        _, answer = bot.split_reasoning("answer is 42<thi")
        self.assertEqual(answer, "answer is 42")

    def test_literal_lt_released_when_not_a_tag(self):
        _, answer = bot.split_reasoning("a < b")
        self.assertEqual(answer, "a < b")

    def test_trailing_marker_from_truncated_stream_is_removed(self):
        _, answer = bot.split_reasoning("answer is 42</thi")
        self.assertEqual(answer, "answer is 42")

    def test_streaming_incremental_matches_final(self):
        raw = "<think>let me think</think>final answer"
        seen = ""
        for i in range(1, len(raw) + 1):
            _, answer = bot.split_reasoning(raw[:i])
            self.assertTrue(answer.startswith(seen))
            seen = answer
        self.assertEqual(seen, "final answer")

    def test_thinking_chunks_never_leak(self):
        raw = "<think>secret plan</think>visible"
        for i in range(1, len(raw) + 1):
            _, answer = bot.split_reasoning(raw[:i])
            self.assertNotIn("secret", answer)


class ChunkTests(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(bot.chunk_text("hi"), ["hi"])

    def test_empty_text(self):
        self.assertEqual(bot.chunk_text(""), [""])

    def test_long_text_split_within_limit(self):
        text = "x" * 10_000
        chunks = bot.chunk_text(text, limit=4000)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c) <= 4000 for c in chunks))
        self.assertEqual("".join(chunks), text)

    def test_prefers_newline_boundary(self):
        text = "a" * 3000 + "\n" + "b" * 3000
        chunks = bot.chunk_text(text, limit=4000)
        self.assertTrue(chunks[0].endswith("\n"))

    def test_chunks_are_stable_while_streaming(self):
        text = "line of text\n" * 800
        first = bot.chunk_text(text, limit=4000)
        second = bot.chunk_text(text + "tail", limit=4000)
        self.assertEqual(first[:-1], second[: len(first) - 1])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = bot.Store(Path(self.tmp.name) / "test.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_history_roundtrip(self):
        self.store.add_message(1, "user", "hello")
        self.store.add_message(1, "assistant", "hi")
        self.assertEqual(
            self.store.history(1, 10, 1000), [("user", "hello"), ("assistant", "hi")]
        )

    def test_history_is_chat_scoped(self):
        self.store.add_message(1, "user", "one")
        self.store.add_message(2, "user", "two")
        self.assertEqual(self.store.history(2, 10, 1000), [("user", "two")])

    def test_history_starts_on_user_turn(self):
        self.store.add_message(1, "user", "a")
        self.store.add_message(1, "assistant", "b")
        self.store.add_message(1, "user", "c")
        rows = self.store.history(1, 2, 1000)
        self.assertEqual(rows[0][0], "user")

    def test_char_budget_trims_oldest(self):
        self.store.add_message(1, "user", "x" * 500)
        self.store.add_message(1, "assistant", "y" * 500)
        self.store.add_message(1, "user", "z" * 500)
        rows = self.store.history(1, 10, 700)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0][1].startswith("z"))

    def test_clear(self):
        self.store.add_message(1, "user", "hello")
        self.assertEqual(self.store.clear_messages(1), 1)
        self.assertEqual(self.store.count_messages(1), 0)

    def test_claim_owner_is_first_wins(self):
        self.assertEqual(self.store.claim_owner(11), 11)
        self.assertEqual(self.store.claim_owner(22), 11)

    def test_state_roundtrip(self):
        self.store.set_state("model", "abc")
        self.assertEqual(self.store.get_state("model"), "abc")


class LiveOllamaTests(unittest.TestCase):
    """Hits the real Ollama daemon; skipped when it isn't running."""

    def setUp(self):
        bot.load_env_file(Path(__file__).resolve().parent.parent / ".env")
        self.cfg = bot.Config()
        try:
            self.models = bot.ollama_tags(self.cfg.ollama_url)
        except bot.OllamaError:
            self.skipTest("ollama not reachable")

    def test_model_is_installed(self):
        self.assertIn(self.cfg.model, self.models)

    def test_streaming_returns_an_answer(self):
        pieces = []
        stats = bot.ollama_chat(
            self.cfg.ollama_url,
            self.cfg.model,
            [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            self.cfg,
            pieces.append,
            threading.Event(),
        )
        thinking, answer = bot.split_reasoning("".join(pieces))
        self.assertTrue(stats.get("eval_count", 0) > 0)
        self.assertIn("4", answer)


class FakeTelegram:
    """Records what would have been sent to Telegram."""

    def __init__(self):
        self.sent: list[tuple[str, str]] = []
        self.edits: list[tuple[int, str]] = []
        self._next_id = 100

    def send_message(self, chat_id, text):
        self.sent.append((chat_id, text))
        self._next_id += 1
        return {"message_id": self._next_id}

    def edit_message(self, chat_id, message_id, text):
        self.edits.append((message_id, text))

    def send_typing(self, chat_id):
        pass


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        bot.load_env_file(Path(__file__).resolve().parent.parent / ".env")
        self.cfg = bot.Config()
        try:
            bot.ollama_tags(self.cfg.ollama_url)
        except bot.OllamaError:
            self.skipTest("ollama not reachable")
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg.data_dir = Path(self.tmp.name)
        self.cfg.log_file = str(Path(self.tmp.name) / "bot.log")
        self.app = bot.App(self.cfg)
        self.fake = FakeTelegram()
        self.app.tg = self.fake

    def tearDown(self):
        self.app.pool.shutdown(wait=True)
        self.tmp.cleanup()

    def test_answer_streams_edits_and_persists(self):
        self.app.answer(555, "What is 2+2? Answer with just the number.")
        final = self.fake.edits[-1][1]
        self.assertIn("4", final)
        self.assertNotIn("<think>", final)
        self.assertNotIn("</think>", final)
        self.assertGreaterEqual(len(self.fake.edits), 1)
        self.assertEqual(
            [role for role, _ in self.app.store.history(555, 10, 10_000)],
            ["user", "assistant"],
        )

    def test_history_is_sent_back_to_the_model(self):
        self.app.store.add_message(556, "user", "my name is Ada")
        self.app.store.add_message(556, "assistant", "noted")
        messages = self.app.build_messages(556, "what is my name?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("my name is Ada", [m["content"] for m in messages])
        self.assertEqual(messages[-1], {"role": "user", "content": "what is my name?"})

    def test_show_thinking_includes_reasoning_block(self):
        self.app.store.set_state("think:557", "1")
        self.app.answer(557, "What is 2+2? Answer with just the number.")
        final = self.fake.edits[-1][1]
        self.assertIn("🧠 reasoning", final)


class SeparateThinkingFieldTests(unittest.TestCase):
    """Ollama puts reasoning in message.thinking for thinking models.

    Streaming used to wrap each of those chunks in a <think> tag, which never
    got closed and swallowed the whole answer.
    """

    def _serve(self, chunks):
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        body = ("\n".join(json.dumps(chunk) for chunk in chunks) + "\n").encode()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("content-type", "application/x-ndjson")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def test_answer_survives_separate_reasoning_chunks(self):
        chunks = [
            {"message": {"role": "assistant", "content": "", "thinking": "Let me think. "}, "done": False},
            {"message": {"role": "assistant", "content": "", "thinking": "Paris."}, "done": False},
            {"message": {"role": "assistant", "content": "The capital is Paris."}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "eval_count": 12,
                "eval_duration": 1_000_000_000,
                "done_reason": "stop",
            },
        ]
        server = self._serve(chunks)
        cfg = bot.Config()
        cfg.timeout = 10
        pieces: list[str] = []
        try:
            stats = bot.ollama_chat(
                f"http://127.0.0.1:{server.server_port}",
                "test-model",
                [{"role": "user", "content": "capital of France?"}],
                cfg,
                pieces.append,
                threading.Event(),
            )
        finally:
            server.shutdown()
            server.server_close()

        thinking, answer = bot.split_reasoning("".join(pieces))
        self.assertEqual(answer.strip(), "The capital is Paris.")
        self.assertEqual(thinking, "")
        self.assertIn("Let me think", stats["thinking"])
        self.assertEqual(stats["done_reason"], "stop")

    def test_answer_survives_inline_think_tags(self):
        chunks = [
            {"message": {"content": "<think>reasoning"}, "done": False},
            {"message": {"content": " here</think>\nThe answer."}, "done": False},
            {"message": {"content": ""}, "done": True, "eval_count": 5},
        ]
        server = self._serve(chunks)
        cfg = bot.Config()
        cfg.timeout = 10
        pieces: list[str] = []
        try:
            bot.ollama_chat(
                f"http://127.0.0.1:{server.server_port}",
                "test-model",
                [{"role": "user", "content": "hi"}],
                cfg,
                pieces.append,
                threading.Event(),
            )
        finally:
            server.shutdown()
            server.server_close()
        thinking, answer = bot.split_reasoning("".join(pieces))
        self.assertEqual(answer.strip(), "The answer.")
        self.assertIn("reasoning here", thinking)


if __name__ == "__main__":
    unittest.main(verbosity=2)

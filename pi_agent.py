"""Run the `pi` coding agent as a subprocess and stream its JSON events to Telegram.

Pi owns the agent loop, tools, sessions and compaction. The bot only:
  * launches `pi --mode json ...`
  * renders text deltas, tool calls and errors into chat messages
  * answers Pi's approval requests (via the minamo-approval extension) with
    Telegram buttons
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from config import Config
from llm import clamp
from store import Store
from telegram_client import StreamRenderer, Telegram, TelegramError, keyboard
from tools import Workspace

log = logging.getLogger("minamo.pi")

READ_ONLY = {"read", "ls", "find", "grep", "tool_search"}


def pi_models_path() -> Path:
    return Path(os.environ.get("PI_MODELS_FILE", str(Path.home() / ".pi/agent/models.json")))


def pi_model_ids() -> list[str]:
    """Model IDs pi knows about, from ~/.pi/agent/models.json."""
    try:
        config = json.loads(pi_models_path().read_text())
    except (OSError, ValueError):
        return []
    ids: list[str] = []
    for provider in (config.get("providers") or {}).values():
        if not isinstance(provider, dict):
            continue
        for model in provider.get("models") or []:
            if isinstance(model, dict) and model.get("id"):
                ids.append(str(model["id"]))
    return ids


def ensure_pi_model(
    model_id: str, context_window: int = 32768, max_tokens: int = 4096
) -> bool:
    """Register an Ollama model in pi's models.json so /agentmodel can use it."""
    path = pi_models_path()
    try:
        config = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    providers = config.setdefault("providers", {})
    ollama = providers.setdefault(
        "ollama",
        {
            "api": "openai-completions",
            "apiKey": "ollama",
            "baseUrl": "http://127.0.0.1:11434/v1",
            "models": [],
        },
    )
    models = ollama.setdefault("models", [])
    if any(isinstance(model, dict) and model.get("id") == model_id for model in models):
        return False
    models.append(
        {
            "id": model_id,
            "name": model_id,
            "reasoning": True,
            "input": ["text"],
            "contextWindow": context_window,
            "maxTokens": max_tokens,
        }
    )
    try:
        path.write_text(json.dumps(config, indent=2) + "\n")
    except OSError as exc:
        log.warning("cannot update %s: %s", path, exc)
        return False
    return True


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
    return "".join(parts)


def result_text(result) -> str:
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return str(result)
    content = result.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    parts.append("[image]")
        if parts:
            return "\n".join(parts)
    return json.dumps(result)[:600]


def short_args(tool: str, args) -> str:
    if not isinstance(args, dict):
        return str(args)[:120]
    if tool == "bash":
        return str(args.get("command", ""))[:400]
    if tool in ("write", "edit", "read", "ls"):
        return str(args.get("path", args.get("pattern", "")))[:200]
    return json.dumps(args)[:200]


def describe_approval(tool: str, args) -> str:
    if tool == "bash":
        return f"🔐 pi wants to run:\n```\n{str(args.get('command', ''))[:1200]}\n```"
    if tool == "write":
        content = str(args.get("content", ""))
        preview = content[:500] + ("…" if len(content) > 500 else "")
        return f"🔐 pi wants to write {args.get('path')} ({len(content)} bytes)\n```\n{preview}\n```"
    if tool == "edit":
        return f"🔐 pi wants to edit {args.get('path')}"
    return f"🔐 pi wants to call {tool}\n```\n{json.dumps(args)[:800]}\n```"


class PiRunner:
    def __init__(self, cfg: Config, store: Store, tg: Telegram, workspace: Workspace) -> None:
        self.cfg = cfg
        self.store = store
        self.tg = tg
        self.workspace = workspace
        self._procs: dict[int, subprocess.Popen] = {}
        self._cancelled: set[int] = set()
        self._guard = threading.Lock()
        self._locks: dict[int, threading.Lock] = {}
        self._locks_guard = threading.Lock()
        self.pending: dict[str, dict] = {}
        self.pending_lock = threading.Lock()
        self.owner_id: int | None = None

    # -- helpers ------------------------------------------------------------ #
    def lock(self, chat_id: int) -> threading.Lock:
        with self._locks_guard:
            if chat_id not in self._locks:
                self._locks[chat_id] = threading.Lock()
            return self._locks[chat_id]

    def say(self, chat_id: int, text: str, markup: dict | None = None) -> int | None:
        try:
            return self.tg.send_message(chat_id, text[:4000], reply_markup=markup)["message_id"]
        except TelegramError as exc:
            log.warning("pi send failed: %s", exc)
            return None

    def edit(self, chat_id: int, message_id: int | None, text: str, markup=None) -> None:
        if not message_id:
            return
        try:
            self.tg.edit_message(chat_id, message_id, text[:4000], reply_markup=markup)
        except TelegramError as exc:
            log.debug("pi edit failed: %s", exc)

    def model(self, chat_id: int) -> str:
        return self.store.get_state(f"pi_model:{chat_id}") or self.cfg.pi_model

    def session_id(self, chat_id: int) -> str:
        generation = self.store.get_state(f"pi_session:{chat_id}") or "1"
        return f"minamo-{chat_id}-{generation}"

    def reset_session(self, chat_id: int) -> None:
        generation = int(self.store.get_state(f"pi_session:{chat_id}") or "1")
        self.store.set_state(f"pi_session:{chat_id}", str(generation + 1))

    def running(self, chat_id: int) -> bool:
        with self._guard:
            return chat_id in self._procs

    def auto(self, chat_id: int) -> bool:
        stored = self.store.get_state(f"auto:{chat_id}")
        if stored is None:
            return self.cfg.agent_auto
        return stored == "1"

    def cancel(self, chat_id: int) -> bool:
        with self._guard:
            proc = self._procs.get(chat_id)
            self._cancelled.add(chat_id)
        self.deny_pending(chat_id, "Run cancelled")
        if proc is None:
            return False
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                proc.terminate()
            except OSError:
                pass
        return True

    def _timeout_run(self, chat_id: int) -> None:
        log.warning("pi run for chat %s exceeded %ss, killing it", chat_id, self.cfg.pi_timeout)
        self.cancel(chat_id)
        self.say(chat_id, f"⌛ pi hit the {self.cfg.pi_timeout}s limit and was stopped.")

    # -- approvals ---------------------------------------------------------- #
    def deny_pending(self, chat_id: int, reason: str) -> None:
        with self.pending_lock:
            entries = [
                (approval_id, entry)
                for approval_id, entry in self.pending.items()
                if entry["chat_id"] == chat_id
            ]
        for approval_id, entry in entries:
            self._resolve(approval_id, False, reason, "expired")

    def _resolve(self, approval_id: str, allow: bool, reason: str, status: str) -> None:
        row = self.store.get_approval(approval_id)
        if row and row["status"] == "pending":
            self.store.resolve_approval(approval_id, status)
            self.edit(
                row["chat_id"],
                row["message_id"],
                ("✅ approved" if allow else f"🚫 denied — {reason}"),
                {"inline_keyboard": []},
            )
        with self.pending_lock:
            entry = self.pending.get(approval_id)
        if entry:
            entry["allow"] = allow
            entry["reason"] = reason
            entry["event"].set()

    def decide(self, approval_id: str, allow: bool) -> str:
        row = self.store.get_approval(approval_id)
        if row is None or row["status"] != "pending":
            return "already handled"
        self._resolve(approval_id, allow, "denied from Telegram" if not allow else "", "approved" if allow else "denied")
        return "approved" if allow else "denied"

    def request_approval(self, chat_id: int, tool: str, args: dict) -> tuple[bool, str]:
        """Called from the HTTP bridge while Pi is blocked in tool_call."""
        if tool in READ_ONLY:
            return True, "read-only"
        if self.auto(chat_id):
            self.say(chat_id, f"⚡ auto-approved {tool}: {short_args(tool, args)[:300]}")
            return True, "auto-approved"

        approval_id = secrets.token_hex(5)
        self.store.add_approval(
            {
                "id": approval_id,
                "chat_id": chat_id,
                "payload": {"tool": tool, "args": args},
                "status": "pending",
                "created_at": time.time(),
            }
        )
        entry = {
            "chat_id": chat_id,
            "event": threading.Event(),
            "allow": False,
            "reason": "timed out",
        }
        with self.pending_lock:
            self.pending[approval_id] = entry

        message_id = self.say(
            chat_id,
            describe_approval(tool, args),
            markup=keyboard(
                [
                    [("✅ Approve", f"apv:{approval_id}"), ("❌ Deny", f"den:{approval_id}")],
                    [("⚡ Auto-approve this chat", f"apva:{approval_id}")],
                ]
            ),
        )
        if message_id:
            self.store.set_approval_message(approval_id, message_id)

        if not entry["event"].wait(self.cfg.pi_approval_timeout):
            with self.pending_lock:
                self.pending.pop(approval_id, None)
            self.store.resolve_approval(approval_id, "expired")
            self.edit(chat_id, message_id, "⌛ approval timed out", {"inline_keyboard": []})
            return False, "approval timed out"

        with self.pending_lock:
            self.pending.pop(approval_id, None)
        return bool(entry["allow"]), str(entry["reason"])

    # -- run ---------------------------------------------------------------- #
    def start(self, chat_id: int, task: str, cwd: Path) -> None:
        if not self.lock(chat_id).acquire(blocking=False):
            self.say(chat_id, "Already working in this chat. Send /stop first.")
            return
        try:
            self._run(chat_id, task, cwd)
        except Exception:
            log.exception("pi run failed")
            self.say(chat_id, "⚠️ pi crashed — check the bot log.")
        finally:
            self.lock(chat_id).release()

    def _run(self, chat_id: int, task: str, cwd: Path) -> None:
        model = self.model(chat_id)
        argv = [
            self.cfg.pi_bin,
            "--mode",
            "json",
            "--provider",
            self.cfg.pi_provider,
            "--model",
            model,
            "--thinking",
            self.cfg.pi_thinking,
            "--tools",
            self.cfg.pi_tools,
            "--session-id",
            self.session_id(chat_id),
            "--",
            task,
        ]
        env = os.environ.copy()
        env["PATH"] = f"{Path.home() / '.npm-global/bin'}:{env.get('PATH', '')}"
        env["MINAMO_APPROVAL_URL"] = f"http://127.0.0.1:{self.cfg.approval_port}/approve"
        env["MINAMO_CHAT_ID"] = str(chat_id)
        env["MINAMO_APPROVAL_TIMEOUT_MS"] = str(self.cfg.pi_approval_timeout * 1000)

        with self._guard:
            self._cancelled.discard(chat_id)
        self.store.set_state(f"pi_running:{chat_id}", json.dumps({"task": task, "started": time.time()}))
        self.say(chat_id, f"🤖 pi · {model} · {self.cfg.pi_thinking}\n▶ {task[:400]}")

        started = time.time()
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            self.store.delete_state(f"pi_running:{chat_id}")
            self.say(chat_id, f"⚠️ could not start pi: {exc}")
            return

        with self._guard:
            self._procs[chat_id] = proc

        watchdog = threading.Timer(self.cfg.pi_timeout, self._timeout_run, args=(chat_id,))
        watchdog.daemon = True
        watchdog.start()

        stderr_lines: list[str] = []

        def drain() -> None:
            for raw in proc.stderr:  # type: ignore[union-attr]
                line = raw.decode("utf-8", "replace").rstrip()
                if line:
                    stderr_lines.append(line)
                    del stderr_lines[:-100]
                    log.info("pi stderr: %s", line)

        threading.Thread(target=drain, daemon=True).start()

        live_mid = self.say(chat_id, "…")
        renderer = StreamRenderer(self.tg, chat_id, live_mid or 0, self.cfg.edit_interval)
        buffer: list[str] = []
        final_text = ""
        tool_messages: dict[str, int | None] = {}
        error_text = ""

        try:
            for raw in proc.stdout:  # type: ignore[union-attr]
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("type")

                if kind == "message_update":
                    update = event.get("assistantMessageEvent") or {}
                    if update.get("type") == "text_delta":
                        buffer.append(update.get("delta", ""))
                        renderer.render("".join(buffer))
                elif kind == "message_end":
                    message = event.get("message") or {}
                    if message.get("role") == "assistant":
                        text = message_text(message)
                        if text.strip():
                            final_text = text
                            buffer = [text]
                            renderer.render(text)
                elif kind == "tool_execution_start":
                    tool = str(event.get("toolName", "tool"))
                    args = event.get("args") or {}
                    tool_messages[event.get("toolCallId", "")] = self.say(
                        chat_id, f"🔧 {tool}: {short_args(tool, args)}"
                    )
                    buffer = []
                elif kind == "tool_execution_end":
                    call_id = event.get("toolCallId", "")
                    tool = str(event.get("toolName", "tool"))
                    ok = not event.get("isError")
                    body = clamp(result_text(event.get("result")), 700)
                    duration = event.get("durationMs")
                    suffix = f" · {duration / 1000:.1f}s" if duration else ""
                    self.edit(
                        chat_id,
                        tool_messages.get(call_id),
                        f"{'✅' if ok else '❌'} {tool}{suffix}\n{body}",
                    )
                elif kind == "error":
                    error_text = str(event.get("error") or event.get("reason") or "unknown error")
                    log.warning("pi error event: %s", error_text)
                elif kind == "agent_settled":
                    break
        except (OSError, ValueError) as exc:
            error_text = error_text or f"stream error: {exc}"

        code = proc.wait()
        watchdog.cancel()
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        with self._guard:
            self._procs.pop(chat_id, None)
            cancelled = chat_id in self._cancelled
            self._cancelled.discard(chat_id)
        self.store.delete_state(f"pi_running:{chat_id}")

        elapsed = time.time() - started
        if cancelled:
            output = "[stopped]"
        elif final_text.strip():
            output = final_text.strip()
            if code != 0:
                output += f"\n\n(pi exited {code})"
        elif error_text:
            output = f"⚠️ {error_text}"
        elif code != 0:
            tail = "\n".join(stderr_lines[-6:])[-1200:]
            output = f"⚠️ pi exited {code}\n{tail}"
        else:
            output = "(pi finished without a final message)"

        renderer.render(output, force=True)
        if live_mid and not renderer.sent:
            self.edit(chat_id, live_mid, output)
        self.say(chat_id, f"⏱ {elapsed:.1f}s · {model} · session {self.session_id(chat_id)}")
        log.info("pi run chat=%s code=%s in %.1fs", chat_id, code, elapsed)


class ApprovalServer:
    """Local HTTP endpoint the Pi extension calls before changing anything."""

    def __init__(self, runner: PiRunner, port: int) -> None:
        self.runner = runner
        self.port = port
        self.httpd: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def start(self, host: str = "127.0.0.1") -> bool:
        runner = self.runner

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                if self.path != "/approve":
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("content-length", 0))
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except (ValueError, OSError):
                    self._reply({"allow": False, "reason": "bad request"})
                    return

                chat_id = int(body.get("chat_id") or 0)
                if runner.owner_id is not None and chat_id != runner.owner_id:
                    self._reply({"allow": False, "reason": "unknown chat"})
                    return
                tool = str(body.get("tool") or "")
                args = body.get("args") if isinstance(body.get("args"), dict) else {}
                allow, reason = runner.request_approval(chat_id, tool, args)
                self._reply({"allow": allow, "reason": reason})

            def _reply(self, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except OSError:
                    pass

            def log_message(self, *args):  # silence default stderr logging
                return

        try:
            self.httpd = ThreadingHTTPServer((host, self.port), Handler)
        except OSError as exc:
            log.error("cannot bind approval port %s: %s", self.port, exc)
            return False
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, name="approval-bridge", daemon=True
        )
        self.thread.start()
        log.info("approval bridge listening on %s:%s", host, self.port)
        return True

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

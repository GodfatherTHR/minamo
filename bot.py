#!/usr/bin/env python3
"""Minamo - a durable Telegram bridge to a local Ollama model, with a
remote-build agent for the machine it runs on.

Standard library only, so it runs the same on Ubuntu and macOS.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import llm
from config import APP_DIR, Config, load_env_file
from llm import (  # re-exported for tests and callers
    OllamaError,
    chunk_text,
    clamp,
    ollama_chat,
    ollama_tags,
    split_reasoning,
)
from pi_agent import ApprovalServer, PiRunner, ensure_pi_model, pi_model_ids
from store import Store
from telegram_client import (
    RateLimited,
    StreamRenderer,
    Telegram,
    TelegramError,
    keyboard,
)
from tools import Denied, JobManager, Workspace, check_command, run_blocking, truncate_output

log = logging.getLogger("minamo")

TG_LIMIT = llm.TG_LIMIT


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
HELP_TEXT = """Chat
  ▸ any message        ask the local model
  /new                 clear history + start a fresh pi session
  /stop                stop the reply or the pi run
  /think on|off        show the model's reasoning

Build on this machine
  ▸ !<command>         run a shell command now (e.g. !git status)
  /bg <command>        run in the background, notify when done
  /jobs                background jobs · /log <id> · /kill <id>
  /write <path>        then the file content on following lines
  /cat <path>          show a file
  /ls [path]           list a directory
  /cd [path]           change directory (default ~)

pi agent (handles the task; risky tools need your tap)
  ▸ @<task>            e.g. @find why the tests fail and fix it
  /auto on|off         skip approvals for this chat
  /agentmodel [name]   show/switch the pi model

Other
  /status  /model  /models  /id  /help"""


class App:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.tg = Telegram(cfg.token)
        self.store = Store(cfg.data_dir / "bot.db")
        self.workspace = Workspace(cfg.workspace_root)
        self.jobs = JobManager(cfg.data_dir, self.workspace)
        self.pi = PiRunner(cfg, self.store, self.tg, self.workspace)
        self.approval_server = ApprovalServer(self.pi, cfg.approval_port)
        self.pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chat")
        self.infer_lock = threading.Lock()
        self.cancel_events: dict[int, threading.Event] = {}
        self.cancel_lock = threading.Lock()
        self.started = time.time()
        self.stop_event = threading.Event()
        self._notice_lock = threading.Lock()
        self._last_notice: dict[int, float] = {}

        self.owner_id = int(cfg.owner_chat_id) if cfg.owner_chat_id.lstrip("-").isdigit() else None
        if self.owner_id is None:
            stored = self.store.get_state("owner_chat_id")
            if stored and stored.lstrip("-").isdigit():
                self.owner_id = int(stored)
        self.pi.owner_id = self.owner_id

    # -- helpers ------------------------------------------------------------ #
    def model(self) -> str:
        return self.store.get_state("model") or self.cfg.model

    def show_thinking(self, chat_id: int) -> bool:
        return self.store.get_state(f"think:{chat_id}") == "1"

    def cwd(self, chat_id: int) -> Path:
        stored = self.store.get_state(f"cwd:{chat_id}")
        if stored:
            try:
                return self.workspace.resolve(stored)
            except Denied:
                pass
        return self.workspace.root

    def system_prompt(self) -> str:
        now = datetime.now().astimezone()
        return (
            f"{self.cfg.system_prompt}\n"
            f"Current local time: {now.strftime('%Y-%m-%d %H:%M %Z')}.\n"
            f"The machine has a bot that can run commands when the owner sends !cmd."
        )

    def build_messages(self, chat_id: int, user_text: str) -> list[dict]:
        messages = [{"role": "system", "content": self.system_prompt()}]
        for role, content in self.store.history(
            chat_id, self.cfg.history_messages, self.cfg.history_chars
        ):
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        return messages

    def safe_send(self, chat_id: int, text: str, markup: dict | None = None) -> int | None:
        try:
            return self.tg.send_message(chat_id, text[:TG_LIMIT], reply_markup=markup)[
                "message_id"
            ]
        except TelegramError as exc:
            log.warning("sendMessage failed: %s", exc)
            return None

    def safe_edit(self, chat_id: int, message_id: int | None, text: str, markup=None) -> None:
        if not message_id:
            return
        try:
            self.tg.edit_message(chat_id, message_id, text[:TG_LIMIT], reply_markup=markup)
        except TelegramError as exc:
            log.debug("editMessage failed: %s", exc)

    # -- update loop -------------------------------------------------------- #
    def run(self) -> None:
        me = self.tg.get_me()
        log.info("connected as @%s (bot id %s)", me.get("username"), me.get("id"))
        self.recover()
        self.approval_server.start()
        threading.Thread(target=self.monitor_jobs, daemon=True).start()
        offset = int(self.store.get_state("update_offset") or 0)
        log.info("polling for updates from offset %s", offset)
        while not self.stop_event.is_set():
            try:
                updates = self.tg.get_updates(offset)
            except TelegramError as exc:
                log.warning("getUpdates failed: %s", exc)
                time.sleep(3)
                continue
            for update in updates:
                offset = max(offset, int(update["update_id"]) + 1)
                self.store.set_state("update_offset", str(offset))
                try:
                    self.dispatch(update)
                except Exception:  # never let one bad update kill the bot
                    log.exception("dispatch failed")
        log.info("shutting down")

    def recover(self) -> None:
        """Clean up state left behind by a restart."""
        for approval in self.store.pending_approvals():
            self.store.resolve_approval(approval["id"], "expired")
        for job in self.store.running_jobs():
            log.info("resuming tracking of job %s", job["id"])
        for key, value in self.store.states_with_prefix("pi_running:"):
            chat_id = int(key.split(":", 1)[1])
            self.store.delete_state(key)
            try:
                info = json.loads(value)
            except ValueError:
                info = {}
            task = str(info.get("task", ""))[:300]
            try:
                self.tg.send_message(
                    chat_id,
                    f"♻️ The bot restarted during this pi run, so it stopped:\n{task}\n\n"
                    "Send the task again if you still want it.",
                )
            except TelegramError as exc:
                log.warning("recover notice failed: %s", exc)

    def dispatch(self, update: dict) -> None:
        if "callback_query" in update:
            self.handle_callback(update["callback_query"])
            return
        message = update.get("message")
        if not message:
            return
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is None:
            return
        self.store.touch_chat(chat)

        text = (message.get("text") or "").strip()
        if not text:
            if chat_id == self.owner_id:
                self.safe_send(chat_id, "I can only read text for now.")
            return
        if chat.get("type") != "private" and chat_id != self.owner_id:
            return
        self.handle_message(chat_id, text)

    def handle_message(self, chat_id: int, text: str) -> None:
        command = text.split(maxsplit=1)[0].split("@")[0].lower()

        if self.owner_id is None:
            if command in ("/start", "/claim"):
                owner = self.store.claim_owner(chat_id)
                self.owner_id = owner
                if owner == chat_id:
                    log.info("owner bound to chat %s", chat_id)
                    self.safe_send(
                        chat_id,
                        "Locked to this chat as owner.\n\n"
                        "Chat with me, or run `!<command>` to execute shell commands, "
                        "`/bg <command>` for long builds, or `@<task>` to have the model "
                        "work the machine with your approval.\n\n/help for everything.",
                    )
                else:
                    self.safe_send(chat_id, "This bot already belongs to another chat.")
                return
            self.safe_send(
                chat_id,
                "This bot isn't set up yet. Send /start to claim it as its owner "
                "(or set OWNER_CHAT_ID in .env).",
            )
            return

        if chat_id != self.owner_id:
            with self._notice_lock:
                last = self._last_notice.get(chat_id, 0)
                if time.time() - last > 3600:
                    self._last_notice[chat_id] = time.time()
                    self.safe_send(chat_id, "This is a private bot.")
            return

        # direct shell
        if text.startswith("!"):
            self.shell(chat_id, text[1:].strip())
            return
        # pi agent
        if text.startswith("@"):
            task = text[1:].strip()
            if not task:
                self.safe_send(chat_id, "Give pi a task: @find why the tests fail and fix it")
                return
            self.pool.submit(self.pi.start, chat_id, task, self.cwd(chat_id))
            return

        if command in ("/start", "/help"):
            self.safe_send(chat_id, HELP_TEXT)
        elif command in ("/new", "/reset", "/clear"):
            removed = self.store.clear_messages(chat_id)
            self.pi.reset_session(chat_id)
            self.safe_send(chat_id, f"Fresh conversation. Cleared {removed} messages.")
        elif command == "/stop":
            self.cancel(chat_id)
            stopped = self.pi.cancel(chat_id)
            self.safe_send(chat_id, "Stopping pi…" if stopped else "Stopping.")
        elif command == "/id":
            self.safe_send(chat_id, f"chat id: {chat_id}")
        elif command == "/agentmodel":
            self.agent_model(chat_id, text)
        elif command == "/auto":
            parts = text.split(maxsplit=1)
            if len(parts) == 1:
                state = "on" if self.pi.auto(chat_id) else "off"
                self.safe_send(chat_id, f"Auto-approve is {state}. /auto on to skip approvals.")
                return
            value = parts[1].strip().lower()
            if value not in ("on", "off"):
                self.safe_send(chat_id, "Use /auto on or /auto off.")
                return
            self.store.set_state(f"auto:{chat_id}", "1" if value == "on" else "0")
            self.safe_send(chat_id, f"Auto-approve {value}.")
        elif command == "/cd":
            self.change_dir(chat_id, text)
        elif command == "/ls":
            path = text.split(maxsplit=1)[1] if len(text.split(maxsplit=1)) > 1 else "."
            self.list_dir(chat_id, path)
        elif command == "/cat":
            parts = text.split(maxsplit=1)
            if len(parts) == 1:
                self.safe_send(chat_id, "Usage: /cat <path>")
                return
            self.show_file(chat_id, parts[1].strip())
        elif command == "/write":
            self.write_file(chat_id, text)
        elif command == "/bg":
            parts = text.split(maxsplit=1)
            if len(parts) == 1:
                self.safe_send(chat_id, "Usage: /bg <command>")
                return
            self.start_job(chat_id, parts[1].strip())
        elif command == "/jobs":
            self.list_jobs(chat_id)
        elif command == "/log":
            self.show_log(chat_id, text)
        elif command == "/kill":
            self.kill_job(chat_id, text)
        elif command == "/models":
            self.list_models(chat_id)
        elif command == "/model":
            self.switch_model(chat_id, text)
        elif command == "/think":
            self.toggle_thinking(chat_id, text)
        elif command == "/status":
            self.safe_send(chat_id, self.status_text(chat_id))
        elif text.startswith("/"):
            self.safe_send(chat_id, "Unknown command. /help lists what I can do.")
        else:
            if len(text) > self.cfg.max_input_chars:
                text = text[: self.cfg.max_input_chars] + "\n\n[truncated]"
            self.pool.submit(self.answer, chat_id, text)

    # -- callbacks ---------------------------------------------------------- #
    def handle_callback(self, callback: dict) -> None:
        data = callback.get("data") or ""
        chat_id = (callback.get("message") or {}).get("chat", {}).get("id")
        callback_id = callback.get("id")
        self.tg.answer_callback(callback_id or "", "")
        if chat_id is None or chat_id != self.owner_id:
            return

        if data.startswith(("apv:", "den:", "apva:")):
            kind, _, approval_id = data.partition(":")
            approve = kind in ("apv", "apva")
            if kind == "apva":
                self.store.set_state(f"auto:{chat_id}", "1")
            result = self.pi.decide(approval_id, approve)
            log.info("approval %s -> %s", approval_id, result)
            return
        if data.startswith("kjob:"):
            job_id = data.split(":", 1)[1]
            job = self.store.get_job(job_id)
            if job is None:
                self.safe_send(chat_id, "Unknown job.")
                return
            message = self.jobs.kill(job["id"], job["unit"], job["pid"])
            self.store.update_job(
                job["id"], status="killed", finished_at=time.time(), exit_code=job["exit_code"]
            )
            self.safe_send(chat_id, f"🛑 job {job['id']} {message}")
            return

    # -- shell / files ------------------------------------------------------ #
    def shell(self, chat_id: int, command: str) -> None:
        if not command:
            self.safe_send(chat_id, "Usage: !<command>  e.g. !git status")
            return
        try:
            check_command(command)
        except Denied as exc:
            self.safe_send(chat_id, f"🚫 {exc}")
            return
        message_id = self.safe_send(chat_id, f"$ {command}\n\n⏳ running…")
        output, code, timed_out = run_blocking(
            command, self.cwd(chat_id), self.cfg.shell_timeout
        )
        body = truncate_output(output)
        if timed_out:
            suffix = f"\n\n⏱ timed out after {self.cfg.shell_timeout}s — use /bg for long jobs"
        else:
            suffix = f"\n\n[exit {code}]"
        self.safe_edit(chat_id, message_id, f"$ {command}\n\n{body}{suffix}")

    def change_dir(self, chat_id: int, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            self.safe_send(chat_id, f"cwd: {self.workspace.display(self.cwd(chat_id))}")
            return
        try:
            target = self.workspace.resolve(parts[1].strip(), self.cwd(chat_id))
        except Denied as exc:
            self.safe_send(chat_id, f"🚫 {exc}")
            return
        if not target.is_dir():
            self.safe_send(chat_id, f"Not a directory: {target}")
            return
        self.store.set_state(f"cwd:{chat_id}", str(target))
        self.safe_send(chat_id, f"cwd: {self.workspace.display(target)}")

    def list_dir(self, chat_id: int, path: str) -> None:
        try:
            target = self.workspace.resolve(path, self.cwd(chat_id))
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))[:200]
        except Denied as exc:
            self.safe_send(chat_id, f"🚫 {exc}")
            return
        except OSError as exc:
            self.safe_send(chat_id, f"⚠️ {exc}")
            return
        lines = [f"{'d' if e.is_dir() else 'f'} {e.name}" for e in entries]
        self.safe_send(chat_id, f"{target}\n" + ("\n".join(lines) or "(empty)"))

    def show_file(self, chat_id: int, path: str) -> None:
        try:
            target = self.workspace.resolve(path, self.cwd(chat_id))
            text = target.read_text(encoding="utf-8", errors="replace")
        except Denied as exc:
            self.safe_send(chat_id, f"🚫 {exc}")
            return
        except OSError as exc:
            self.safe_send(chat_id, f"⚠️ {exc}")
            return
        for chunk in chunk_text(text):
            self.safe_send(chat_id, chunk)

    def write_file(self, chat_id: int, text: str) -> None:
        parts = text.split("\n", 1)
        header = parts[0].split(maxsplit=1)
        if len(header) < 2:
            self.safe_send(
                chat_id,
                "Usage:\n/write path/to/file\n<content on the following lines>",
            )
            return
        path_arg = header[1].strip()
        content = parts[1] if len(parts) > 1 else ""
        content = strip_fence(content)
        if not content.strip():
            self.safe_send(chat_id, "No content. Put the file body after the first line.")
            return
        try:
            target = self.workspace.resolve(path_arg, self.cwd(chat_id))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except Denied as exc:
            self.safe_send(chat_id, f"🚫 {exc}")
            return
        except OSError as exc:
            self.safe_send(chat_id, f"⚠️ {exc}")
            return
        self.safe_send(chat_id, f"✍️ wrote {len(content)} bytes to {target}")

    # -- background jobs ---------------------------------------------------- #
    def start_job(self, chat_id: int, command: str) -> None:
        try:
            job = self.jobs.start(command, self.cwd(chat_id))
        except Denied as exc:
            self.safe_send(chat_id, f"🚫 {exc}")
            return
        except (OSError, ValueError) as exc:
            self.safe_send(chat_id, f"⚠️ could not start job: {exc}")
            return
        job["chat_id"] = chat_id
        self.store.add_job(job)
        runner = "systemd" if job["unit"] else f"setsid pid {job['pid']}"
        self.safe_send(
            chat_id,
            f"🏗 job {job['id']} started ({runner})\n$ {command}\n\n"
            f"I'll message you when it finishes. /log {job['id']} to peek.",
            markup=keyboard([[(f"🛑 Stop job {job['id']}", f"kjob:{job['id']}")]]),
        )

    def list_jobs(self, chat_id: int) -> None:
        self.poll_jobs()
        rows = self.store.recent_jobs(chat_id, 10)
        if not rows:
            self.safe_send(chat_id, "No jobs yet. Start one with /bg <command>.")
            return
        lines = []
        for row in rows:
            icon = {"running": "🏗", "done": "✅", "failed": "❌", "killed": "🛑"}.get(
                row["status"], "•"
            )
            lines.append(f"{icon} {row['id']} [{row['status']}] {row['label'][:70]}")
        self.safe_send(chat_id, "\n".join(lines))

    def show_log(self, chat_id: int, text: str) -> None:
        parts = text.split()
        if len(parts) < 2:
            self.safe_send(chat_id, "Usage: /log <job id> [lines]")
            return
        job = self.store.get_job(parts[1])
        if job is None:
            self.safe_send(chat_id, "Unknown job id.")
            return
        lines = 40
        if len(parts) > 2:
            try:
                lines = max(5, min(400, int(parts[2])))
            except ValueError:
                pass
        tail = self.jobs.tail(job["id"], lines)
        self.safe_send(chat_id, f"📜 {job['id']} [{job['status']}]\n\n{clamp(tail)}")

    def kill_job(self, chat_id: int, text: str) -> None:
        parts = text.split()
        if len(parts) < 2:
            self.safe_send(chat_id, "Usage: /kill <job id>")
            return
        job = self.store.get_job(parts[1])
        if job is None:
            self.safe_send(chat_id, "Unknown job id.")
            return
        result = self.jobs.kill(job["id"], job["unit"], job["pid"])
        self.store.update_job(job["id"], status="killed", finished_at=time.time())
        self.safe_send(chat_id, f"🛑 job {job['id']} {result}")

    def poll_jobs(self) -> None:
        for job in self.store.running_jobs():
            status, code = self.jobs.poll(
                job["id"], job["unit"], job["pid"], job["created_at"]
            )
            if status != "running":
                self.store.update_job(
                    job["id"], status=status, exit_code=code, finished_at=time.time()
                )

    def monitor_jobs(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.poll_jobs()
                for job in self.store.unfinished_notifications():
                    icon = "✅" if job["status"] == "done" else "❌"
                    tail = self.jobs.tail(job["id"], 15)
                    self.safe_send(
                        int(job["chat_id"]),
                        f"{icon} job {job['id']} {job['status']}"
                        f" (exit {job['exit_code']})\n$ {job['command'][:200]}\n\n{clamp(tail, 2500)}\n\n"
                        f"/log {job['id']} for more",
                    )
                    self.store.update_job(job["id"], notified=1)
            except Exception:
                log.exception("job monitor failed")
            self.stop_event.wait(5)

    # -- model -------------------------------------------------------------- #
    def list_models(self, chat_id: int) -> None:
        installed: list[str] = []
        ollama_error = ""
        try:
            installed = ollama_tags(self.cfg.ollama_url)
        except OllamaError as exc:
            ollama_error = str(exc)
        pi_models = pi_model_ids()
        chat_model = self.model()
        agent_model = self.pi.model(chat_id)

        lines = []
        if installed:
            lines.append("ollama models:")
            for name in installed:
                marks = []
                if name == chat_model:
                    marks.append("chat")
                if name == agent_model:
                    marks.append("pi")
                suffix = f"  ← {', '.join(marks)}" if marks else ""
                lines.append(f"  {name}{suffix}")
        else:
            lines.append(ollama_error or "no local models")

        extra = [name for name in pi_models if name not in installed]
        if extra:
            lines.append("configured for pi (not local — cloud or unpulled):")
            for name in extra:
                suffix = "  ← pi" if name == agent_model else ""
                lines.append(f"  {name}{suffix}")

        lines.append("")
        lines.append(f"chat model: {chat_model}")
        lines.append(f"pi model:   {agent_model}")
        lines.append("switch: /model <name> · /agentmodel <name>")
        lines.append("cloud models (e.g. beam:cloud) work without a local download.")
        self.safe_send(chat_id, clamp("\n".join(lines), 3900))

    def switch_model(self, chat_id: int, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            self.safe_send(
                chat_id,
                f"Chat model: {self.model()}\n"
                "Switch with /model <name> — /models lists everything.",
            )
            return
        wanted = parts[1].strip()
        if not wanted:
            self.safe_send(chat_id, "Usage: /model <name>")
            return
        try:
            installed = ollama_tags(self.cfg.ollama_url)
        except OllamaError:
            installed = []
        self.store.set_state("model", wanted)
        if wanted in installed:
            self.safe_send(chat_id, f"Chat model: {wanted}.")
        else:
            self.safe_send(
                chat_id,
                f"Chat model: {wanted}.\n"
                "Not installed locally — Ollama will fetch it on the first message "
                "(cloud models like beam:cloud start instantly).",
            )

    def agent_model(self, chat_id: int, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            self.safe_send(
                chat_id,
                f"pi model: {self.pi.model(chat_id)}\n"
                f"thinking: {self.cfg.pi_thinking} · tools: {self.cfg.pi_tools}\n"
                "Switch with /agentmodel <name>. Any Ollama model is registered "
                "with pi automatically.",
            )
            return
        wanted = parts[1].strip()
        if not wanted:
            self.safe_send(chat_id, "Usage: /agentmodel <name>")
            return
        added = ensure_pi_model(wanted)
        self.store.set_state(f"pi_model:{chat_id}", wanted)
        note = " (registered with pi)" if added else ""
        self.safe_send(chat_id, f"pi model: {wanted}{note}.")
        if added:
            log.info("registered %s in pi models.json", wanted)

    def toggle_thinking(self, chat_id: int, text: str) -> None:
        parts = text.split(maxsplit=1)
        if len(parts) == 1:
            state = "on" if self.show_thinking(chat_id) else "off"
            self.safe_send(chat_id, f"Reasoning display is {state}.")
            return
        value = parts[1].strip().lower()
        if value not in ("on", "off"):
            self.safe_send(chat_id, "Use /think on or /think off.")
            return
        self.store.set_state(f"think:{chat_id}", "1" if value == "on" else "0")
        self.safe_send(chat_id, f"Reasoning display {value}.")

    def status_text(self, chat_id: int) -> str:
        uptime = int(time.time() - self.started)
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        try:
            models = ollama_tags(self.cfg.ollama_url)
            ollama_state = f"up ({len(models)} models)"
        except OllamaError:
            ollama_state = "DOWN"
        running = len(self.store.running_jobs())
        return (
            f"model: {self.model()}\n"
            f"pi: {self.pi.model(chat_id)} · {self.cfg.pi_thinking} · "
            f"{'running' if self.pi.running(chat_id) else 'idle'}\n"
            f"ollama: {ollama_state}\n"
            f"context: {self.cfg.num_ctx} · max tokens: {self.cfg.num_predict}\n"
            f"workspace: {self.workspace.root}\n"
            f"cwd: {self.workspace.display(self.cwd(chat_id))}\n"
            f"history: {self.store.count_messages(chat_id)} messages\n"
            f"auto-approve: {'on' if self.pi.auto(chat_id) else 'off'}\n"
            f"jobs running: {running}\n"
            f"uptime: {hours}h {minutes}m {seconds}s"
        )

    def cancel(self, chat_id: int) -> None:
        with self.cancel_lock:
            event = self.cancel_events.get(chat_id)
        if event:
            event.set()

    def chat_budget(self, model: str) -> int:
        """Cloud models are fast, so give their reasoning plenty of room."""
        if model.endswith(":cloud"):
            return max(self.cfg.num_predict, 8192)
        return self.cfg.num_predict

    # -- chat generation ---------------------------------------------------- #
    def answer(self, chat_id: int, user_text: str) -> None:
        cancel = threading.Event()
        with self.cancel_lock:
            self.cancel_events[chat_id] = cancel

        typing_stop = threading.Event()

        def typing_loop() -> None:
            while not typing_stop.is_set():
                self.tg.send_typing(chat_id)
                typing_stop.wait(4.5)

        typing_thread = threading.Thread(target=typing_loop, daemon=True)
        typing_thread.start()

        placeholder: dict | None = None
        try:
            placeholder = self.tg.send_message(chat_id, "…")
        except TelegramError as exc:
            log.warning("could not send placeholder: %s", exc)

        placeholder_id = placeholder["message_id"] if placeholder else 0
        renderer = StreamRenderer(self.tg, chat_id, placeholder_id, self.cfg.edit_interval)
        raw_parts: list[str] = []
        state = {"rendered": ""}

        def on_piece(piece: str) -> None:
            raw_parts.append(piece)
            _, answer_text = split_reasoning("".join(raw_parts))
            if answer_text and answer_text != state["rendered"]:
                state["rendered"] = answer_text
                if placeholder:
                    renderer.render(answer_text)

        final_text = ""
        stats: dict = {}
        error: str | None = None
        try:
            if not self.infer_lock.acquire(blocking=False):
                self.safe_edit(chat_id, placeholder_id, "⏳ queued…")
                while not self.infer_lock.acquire(timeout=0.5):
                    if cancel.is_set():
                        raise RuntimeError("cancelled while queued")
            try:
                if cancel.is_set():
                    raise RuntimeError("cancelled")
                messages = self.build_messages(chat_id, user_text)
                model = self.model()
                budgets = [self.chat_budget(model)]
                if budgets[0] < 8192:
                    budgets.append(min(budgets[0] * 4, 8192))
                budget = budgets[0]
                thinking = ""
                answer_text = ""
                for attempt, budget in enumerate(budgets):
                    if cancel.is_set():
                        raise RuntimeError("cancelled")
                    raw_parts.clear()
                    state["rendered"] = ""
                    stats = ollama_chat(
                        self.cfg.ollama_url,
                        model,
                        messages,
                        self.cfg,
                        on_piece,
                        cancel,
                        num_predict=budget,
                    )
                    thinking, answer_text = split_reasoning("".join(raw_parts))
                    separated = stats.get("thinking") or ""
                    if separated:
                        thinking = "\n".join(part for part in (thinking, separated) if part).strip()
                    if answer_text.strip() or stats.get("cancelled"):
                        break
                    if attempt < len(budgets) - 1:
                        log.info(
                            "chat %s produced no answer at %s tokens (done=%s), retrying with %s",
                            chat_id,
                            budget,
                            stats.get("done_reason"),
                            budgets[attempt + 1],
                        )
                        self.safe_edit(chat_id, placeholder_id, "🧠 thinking further…")
            finally:
                self.infer_lock.release()

            self.store.add_message(chat_id, "user", user_text)

            if stats.get("cancelled"):
                final_text = (answer_text.strip() + "\n\n[stopped]").strip()
            elif not answer_text.strip():
                final_text = (
                    f"No answer — the model used its {budget}-token budget on reasoning "
                    f"(done: {stats.get('done_reason') or 'unknown'}). "
                    "Try a smaller question, a non-thinking model, or raise "
                    "OLLAMA_NUM_PREDICT."
                )
                if thinking:
                    log.warning("no answer; %d chars of reasoning", len(thinking))
            else:
                final_text = answer_text.strip()
                if self.show_thinking(chat_id) and thinking:
                    final_text = f"🧠 reasoning\n{thinking}\n\n———\n\n{final_text}"

            if self.cfg.stats_footer and answer_text.strip():
                eval_count = stats.get("eval_count") or 0
                duration = (stats.get("total_duration") or 0) / 1e9
                if eval_count and duration:
                    rate = eval_count / duration if duration else 0
                    final_text = f"{final_text}\n\n⏱ {duration:.1f}s · {eval_count} tok · {rate:.1f} tok/s"

            if answer_text.strip():
                self.store.add_message(chat_id, "assistant", answer_text.strip())
        except RuntimeError as exc:
            error = str(exc)
            final_text = "[stopped]" if "cancel" in error else f"error: {error}"
        except OllamaError as exc:
            error = str(exc)
            final_text = f"Ollama problem: {exc}"
            log.error("ollama error: %s", exc)
        except Exception as exc:
            error = repr(exc)
            final_text = "Something broke while generating. Check the log."
            log.exception("generation failed")
        finally:
            typing_stop.set()
            with self.cancel_lock:
                self.cancel_events.pop(chat_id, None)

        if placeholder:
            renderer.render(final_text or "(no output)", force=True)
        else:
            self.safe_send(chat_id, final_text or "(no output)")
        if error:
            log.info("chat %s finished with error: %s", chat_id, error)
        else:
            log.info("chat %s answered %d chars", chat_id, len(final_text))

    def shutdown(self) -> None:
        self.stop_event.set()
        self.pool.shutdown(wait=False)


def strip_fence(content: str) -> str:
    text = content.strip("\n")
    if not text.startswith("```"):
        return content
    lines = text.splitlines()
    lines = lines[1:]  # drop opening fence (may carry a language tag)
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# entrypoints
# --------------------------------------------------------------------------- #
def setup_logging(cfg: Config) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(
            RotatingFileHandler(cfg.log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        )
    except OSError as exc:
        print(f"warning: cannot open log file: {exc}", file=sys.stderr)
    logging.basicConfig(
        level=logging.DEBUG if cfg.debug else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        handlers=handlers,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def cli_prompt(cfg: Config, prompt: str, show_thinking: bool) -> int:
    parts: list[str] = []
    messages = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": prompt},
    ]

    def on_piece(piece: str) -> None:
        parts.append(piece)

    started = time.time()
    stats = ollama_chat(cfg.ollama_url, cfg.model, messages, cfg, on_piece, threading.Event())
    thinking, answer = split_reasoning("".join(parts))
    if show_thinking and thinking:
        print(f"--- reasoning ---\n{thinking}\n--- answer ---")
    print(answer.strip() or "(no answer)")
    duration = (stats.get("total_duration") or 0) / 1e9
    print(
        f"\n[{stats.get('eval_count', 0)} tok in {duration:.1f}s, wall {time.time() - started:.1f}s]",
        file=sys.stderr,
    )
    return 0 if answer.strip() else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram bridge for a local Ollama model")
    parser.add_argument("--prompt", help="ask the model once from the terminal and exit")
    parser.add_argument("--show-thinking", action="store_true")
    parser.add_argument("--check", action="store_true", help="verify config and Ollama, then exit")
    args = parser.parse_args()

    load_env_file(APP_DIR / ".env")
    cfg = Config()
    setup_logging(cfg)

    if args.prompt:
        try:
            return cli_prompt(cfg, args.prompt, args.show_thinking)
        except OllamaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    cfg.validate()

    if args.check:
        me = Telegram(cfg.token).get_me()
        names = ollama_tags(cfg.ollama_url)
        print(f"telegram:  @{me.get('username')} (id {me.get('id')})")
        print(f"ollama:    {cfg.ollama_url} · {len(names)} models")
        print(f"chat:      {cfg.model}")
        print(f"pi:        {cfg.pi_bin}")
        print(f"pi model:  {cfg.pi_model} [{cfg.pi_thinking}] · {'known' if cfg.pi_model in pi_model_ids() else 'NOT in models.json'}")
        print(f"pi tools:  {cfg.pi_tools}")
        print(f"approval:  127.0.0.1:{cfg.approval_port}")
        print(f"workspace: {cfg.workspace_root}")
        print(f"data:      {cfg.data_dir}")
        return 0

    app = App(cfg)

    def handle_signal(signum, _frame) -> None:
        log.info("received signal %s", signum)
        app.shutdown()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while not app.stop_event.is_set():
        try:
            app.run()
        except TelegramError as exc:
            log.error("telegram failure: %s", exc)
            time.sleep(5)
        except KeyboardInterrupt:
            app.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

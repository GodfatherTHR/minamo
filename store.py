"""SQLite store: chat history, bot state, background jobs, approvals, agent sessions."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path


class Store:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._local = threading.local()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chats (
                chat_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                command TEXT NOT NULL,
                cwd TEXT NOT NULL,
                label TEXT,
                log_path TEXT NOT NULL,
                unit TEXT,
                pid INTEGER,
                status TEXT NOT NULL,
                exit_code INTEGER,
                created_at REAL NOT NULL,
                finished_at REAL,
                notified INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE TABLE IF NOT EXISTS approvals (
                id TEXT PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                message_id INTEGER,
                created_at REAL NOT NULL
            );            """
        )
        conn.commit()

    # -- state ------------------------------------------------------------- #
    def get_state(self, key: str) -> str | None:
        row = self._conn().execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_state(self, key: str, value: str) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO state(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()

    def delete_state(self, key: str) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM state WHERE key=?", (key,))
        conn.commit()

    def states_with_prefix(self, prefix: str) -> list[tuple[str, str]]:
        rows = self._conn().execute(
            "SELECT key, value FROM state WHERE key LIKE ?", (prefix.replace("%", "") + "%",)
        ).fetchall()
        return [(row["key"], row["value"]) for row in rows]

    def claim_owner(self, chat_id: int) -> int:
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT value FROM state WHERE key='owner_chat_id'").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO state(key, value) VALUES('owner_chat_id', ?)", (str(chat_id),)
            )
            conn.commit()
            return chat_id
        conn.commit()
        return int(row["value"])

    # -- messages ---------------------------------------------------------- #
    def add_message(self, chat_id: int, role: str, content: str) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO messages(chat_id, role, content, created_at) VALUES(?,?,?,?)",
            (chat_id, role, content, time.time()),
        )
        conn.commit()

    def clear_messages(self, chat_id: int) -> int:
        conn = self._conn()
        cur = conn.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        conn.commit()
        return cur.rowcount

    def count_messages(self, chat_id: int) -> int:
        row = self._conn().execute(
            "SELECT COUNT(*) AS n FROM messages WHERE chat_id=?", (chat_id,)
        ).fetchone()
        return int(row["n"])

    def history(self, chat_id: int, limit: int, max_chars: int) -> list[tuple[str, str]]:
        rows = self._conn().execute(
            "SELECT role, content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?",
            (chat_id, max(1, limit)),
        ).fetchall()
        picked: list[tuple[str, str]] = []
        total = 0
        for row in rows:
            content = row["content"]
            if total + len(content) > max_chars and picked:
                break
            picked.append((row["role"], content))
            total += len(content)
        picked.reverse()
        while picked and picked[0][0] != "user":
            picked.pop(0)
        return picked

    def touch_chat(self, chat: dict) -> None:
        now = time.time()
        conn = self._conn()
        conn.execute(
            "INSERT INTO chats(chat_id, username, first_name, first_seen, last_seen) "
            "VALUES(?,?,?,?,?) ON CONFLICT(chat_id) DO UPDATE SET "
            "username=excluded.username, first_name=excluded.first_name, last_seen=excluded.last_seen",
            (chat.get("id"), chat.get("username"), chat.get("first_name"), now, now),
        )
        conn.commit()

    # -- jobs -------------------------------------------------------------- #
    def add_job(self, job: dict) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO jobs(id, chat_id, command, cwd, label, log_path, unit, pid, status,"
            " created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                job["id"],
                job["chat_id"],
                job["command"],
                job["cwd"],
                job.get("label"),
                job["log_path"],
                job.get("unit"),
                job.get("pid"),
                job["status"],
                job["created_at"],
            ),
        )
        conn.commit()

    def update_job(self, job_id: str, **fields) -> None:
        if not fields:
            return
        assignments = ", ".join(f"{key}=?" for key in fields)
        conn = self._conn()
        conn.execute(
            f"UPDATE jobs SET {assignments} WHERE id=?", (*fields.values(), job_id)
        )
        conn.commit()

    def get_job(self, job_id: str) -> sqlite3.Row | None:
        return self._conn().execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()

    def running_jobs(self) -> list[sqlite3.Row]:
        return self._conn().execute(
            "SELECT * FROM jobs WHERE status='running' ORDER BY created_at"
        ).fetchall()

    def recent_jobs(self, chat_id: int, limit: int = 10) -> list[sqlite3.Row]:
        return self._conn().execute(
            "SELECT * FROM jobs WHERE chat_id=? ORDER BY created_at DESC LIMIT ?",
            (chat_id, limit),
        ).fetchall()

    def unfinished_notifications(self) -> list[sqlite3.Row]:
        return self._conn().execute(
            "SELECT * FROM jobs WHERE status!='running' AND notified=0 ORDER BY finished_at"
        ).fetchall()

    # -- approvals --------------------------------------------------------- #
    def add_approval(self, approval: dict) -> None:
        conn = self._conn()
        conn.execute(
            "INSERT INTO approvals(id, chat_id, payload, status, message_id, created_at)"
            " VALUES(?,?,?,?,?,?)",
            (
                approval["id"],
                approval["chat_id"],
                json.dumps(approval["payload"]),
                approval["status"],
                approval.get("message_id"),
                approval["created_at"],
            ),
        )
        conn.commit()

    def get_approval(self, approval_id: str) -> sqlite3.Row | None:
        return self._conn().execute(
            "SELECT * FROM approvals WHERE id=?", (approval_id,)
        ).fetchone()

    def resolve_approval(self, approval_id: str, status: str) -> None:
        conn = self._conn()
        conn.execute("UPDATE approvals SET status=? WHERE id=?", (status, approval_id))
        conn.commit()

    def set_approval_message(self, approval_id: str, message_id: int) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE approvals SET message_id=? WHERE id=?", (message_id, approval_id)
        )
        conn.commit()

    def pending_approvals(self, chat_id: int | None = None) -> list[sqlite3.Row]:
        if chat_id is None:
            return self._conn().execute(
                "SELECT * FROM approvals WHERE status='pending' ORDER BY created_at"
            ).fetchall()
        return self._conn().execute(
            "SELECT * FROM approvals WHERE status='pending' AND chat_id=? ORDER BY created_at",
            (chat_id,),
        ).fetchall()

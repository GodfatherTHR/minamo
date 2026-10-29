"""Workspace sandbox, command safety, shell execution and durable background jobs."""

from __future__ import annotations

import logging
import os
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

log = logging.getLogger("minamo.tools")

MAX_OUTPUT = 3500


class Denied(Exception):
    """A command or path was refused by policy."""


class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def resolve(self, path: str, cwd: Path | None = None) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = (cwd or self.root) / candidate
        resolved = candidate.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise Denied(f"{path} is outside the workspace ({self.root})")
        return resolved

    def display(self, path: Path) -> str:
        try:
            return "~/" + str(path.relative_to(Path.home()))
        except ValueError:
            return str(path)


# Anything in here is refused even when the owner typed it. These are the
# commands that can destroy the machine before you get a chance to react.
_DENY = [
    (re.compile(r"\brm\s+(-[a-zA-Z-]+\s+)*(/|/\*|~|~/\*|/\s)"), "deleting the filesystem root"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "formatting a filesystem"),
    (re.compile(r"\bdd\b[^\n]*\bof=/dev/(sd|nvme|hd|vd|mmcblk)"), "writing raw disk blocks"),
    (re.compile(r">\s*/dev/(sd|nvme|hd|vd|mmcblk)"), "writing raw disk blocks"),
    (re.compile(r":\s*\(\s*\)\s*\{.*:\s*\|\s*:"), "fork bomb"),
    (re.compile(r"\b(shutdown|reboot|poweroff|halt)\b"), "shutting the machine down"),
    (re.compile(r"\bchmod\s+(-R\s+)?[0-7]{3,4}\s+/(\s|$)"), "changing permissions on /"),
    (re.compile(r"\bchown\s+(-R\s+)?[^\n]*\s+/(\s|$)"), "changing ownership of /"),
    (re.compile(r"\b(curl|wget)\b[^|\n]*\|\s*(sudo\s+)?(ba|z|k)?sh\b"), "piping a download into a shell"),
    (re.compile(r"\bsudo\b"), "sudo (it would block waiting for a password)"),
    (re.compile(r"\bgit\s+push\b[^\n]*--force"), "force-pushing a branch"),
]


def check_command(command: str) -> None:
    text = command.strip()
    if not text:
        raise Denied("empty command")
    for pattern, reason in _DENY:
        if pattern.search(text):
            raise Denied(f"refused: {reason}")
    if len(text) > 4000:
        raise Denied("command is too long (4000 chars)")


def truncate_output(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head = limit // 4
    tail = limit - head - 40
    return f"{text[:head]}\n… [{len(text) - limit} chars cut] …\n{text[-tail:]}"


def run_blocking(command: str, cwd: Path, timeout: int) -> tuple[str, int | None, bool]:
    """Run a command to completion. Returns (output, exit_code, timed_out)."""
    try:
        proc = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
            errors="replace",
        )
        return proc.stdout or "(no output)", proc.returncode, False
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", "replace")
        return partial or "(no output before timeout)", None, True
    except (OSError, ValueError) as exc:
        return f"failed to start command: {exc}", None, False


class JobManager:
    """Background commands that outlive bot restarts.

    On Linux each job runs as a transient systemd user unit, so it is outside
    the bot's cgroup. Where systemd-run is unavailable we fall back to setsid.
    A status file records the exit code, so state survives a bot restart too.
    """

    def __init__(self, data_dir: Path, workspace: Workspace) -> None:
        self.dir = data_dir / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace
        self.use_systemd = self._systemd_works()

    def _systemd_works(self) -> bool:
        if not shutil.which("systemd-run") or not shutil.which("systemctl"):
            return False
        try:
            probe = subprocess.run(
                ["systemctl", "--user", "is-system-running"],
                capture_output=True,
                timeout=5,
            )
            return probe.returncode in (0, 1, 3)  # running / degraded / maintenance
        except (OSError, subprocess.SubprocessError):
            return False

    def paths(self, job_id: str) -> tuple[Path, Path]:
        return self.dir / f"{job_id}.log", self.dir / f"{job_id}.exit"

    def start(self, command: str, cwd: Path, label: str | None = None) -> dict:
        check_command(command)
        job_id = secrets.token_hex(4)
        log_path, exit_path = self.paths(job_id)
        script = f"( {command} ) > {shlex.quote(str(log_path))} 2>&1; echo $? > {shlex.quote(str(exit_path))}"
        unit = None
        pid = None
        status = "running"

        if self.use_systemd:
            unit = f"minamo-job-{job_id}"
            try:
                proc = subprocess.run(
                    [
                        "systemd-run",
                        "--user",
                        "--collect",
                        "--quiet",
                        f"--unit={unit}",
                        f"--working-directory={cwd}",
                        "/bin/bash",
                        "-lc",
                        script,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if proc.returncode != 0:
                    raise OSError(proc.stderr.strip()[:200] or "systemd-run failed")
            except (OSError, subprocess.SubprocessError) as exc:
                log.warning("systemd-run failed, falling back to setsid: %s", exc)
                self.use_systemd = False
                unit = None

        if unit is None:
            with open(log_path, "wb") as out:
                proc = subprocess.Popen(
                    ["/bin/bash", "-lc", script],
                    cwd=str(cwd),
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            pid = proc.pid

        return {
            "id": job_id,
            "command": command,
            "cwd": str(cwd),
            "label": label or command[:60],
            "log_path": str(log_path),
            "unit": unit,
            "pid": pid,
            "status": status,
            "created_at": time.time(),
        }

    # -- status ------------------------------------------------------------ #
    def poll(
        self,
        job_id: str,
        unit: str | None,
        pid: int | None,
        created_at: float | None = None,
    ) -> tuple[str, int | None]:
        """Returns (status, exit_code) with status in running|done|failed.

        The exit file is written by the wrapper before the process ends, so its
        presence is the authoritative signal. A transient unit can briefly look
        inactive while it is still being registered, hence the re-check.
        """
        log_path, exit_path = self.paths(job_id)
        code = self._exit_code(exit_path)
        if code is not None:
            return ("done" if code == 0 else "failed"), code

        if unit:
            state = self._unit_state(unit)
            if state in ("failed", "inactive", "unknown", ""):
                time.sleep(0.4)
                code = self._exit_code(exit_path)
                if code is not None:
                    return ("done" if code == 0 else "failed"), code
                if not log_path.exists():
                    if created_at and time.time() - created_at > 30:
                        return "failed", None  # never produced anything
                    return "running", None  # unit not started yet
                return "failed", None
            return "running", None

        if pid:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return "failed", None
            except PermissionError:
                return "running", None
            return "running", None
        return "failed", None

    @staticmethod
    def _exit_code(exit_path: Path) -> int | None:
        if not exit_path.exists():
            return None
        try:
            return int(exit_path.read_text().strip() or "-1")
        except (OSError, ValueError):
            return -1

    @staticmethod
    def _unit_state(unit: str) -> str:
        try:
            proc = subprocess.run(
                ["systemctl", "--user", "is-active", unit],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return proc.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def tail(self, job_id: str, lines: int = 40) -> str:
        log_path, _ = self.paths(job_id)
        if not log_path.exists():
            return "(no output yet)"
        try:
            with open(log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 65536))
                data = handle.read().decode("utf-8", "replace")
        except OSError as exc:
            return f"(cannot read log: {exc})"
        tail = data.splitlines()[-lines:]
        return "\n".join(tail) if tail else "(empty log)"

    def kill(self, job_id: str, unit: str | None, pid: int | None) -> str:
        killed = False
        if unit:
            try:
                subprocess.run(
                    ["systemctl", "--user", "stop", unit],
                    capture_output=True,
                    timeout=15,
                )
                killed = True
            except (OSError, subprocess.SubprocessError) as exc:
                return f"stop failed: {exc}"
        if pid:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                killed = True
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGTERM)
                    killed = True
                except (ProcessLookupError, PermissionError):
                    pass
        return "stopped" if killed else "process already gone"

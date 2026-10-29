# Minamo

A Telegram control plane for your own machine: chat with a local Ollama model,
run shell commands, and hand real coding tasks to [pi](https://github.com/earendil-works/pi)
— from your phone.

Bring your own bot token. No accounts, no hosted service, no third-party Python
packages. Everything runs on the machine you point it at.

```
Telegram  ⇄  bot.py ─┬─ pi_agent.py ──→ `pi --mode json`   (agent + tools)
                     │                       ↑ approval request over localhost
                     ├─ tools.py             │
                     ├─ store.py             └─ ~/.pi/agent/extensions/minamo-approval.ts
                     └─ llm.py ──→ Ollama (chat only)
```

## What you get

- **Chat** with any Ollama model, streamed into Telegram. Thinking models'
  reasoning is separated and hidden unless you ask for it.
- **`@task` runs the pi coding agent.** Pi owns the agent loop, tools, sessions
  and context compaction; the bot renders its event stream into chat messages.
- **Approvals.** pi's `bash`, `write` and `edit` calls round-trip to Telegram as
  Approve / Deny buttons before anything runs. Read-only tools don't interrupt.
- **Instant commands.** `!git status` runs immediately, no model involved.
- **Durable background jobs.** `/bg npm run build` runs as a transient systemd
  unit, survives bot restarts, keeps a log, and messages you with the exit code.
- **Every model is selectable.** `/models` lists local and cloud Ollama models;
  `/model` and `/agentmodel` switch without editing files.
- **Durable state.** Chat history, jobs, and pending approvals live in SQLite.

## Requirements

| Component | Version | Notes |
| --- | --- | --- |
| Linux or macOS | — | Windows works for the bot, but the service scripts target Linux/macOS |
| Python | 3.11+ | standard library only — no `pip install` |
| [Ollama](https://ollama.com) | recent | local models, or cloud models with an Ollama account |
| Node.js | 22.19+ | required by pi |
| [pi](https://github.com/earendil-works/pi) | 1.x | `curl -fsSL https://pi.dev/install.sh \| sh` |
| Telegram bot | — | create it with [@BotFather](https://t.me/BotFather) |

## 1. Create your bot

1. Open **@BotFather** in Telegram, send `/newbot`, pick a name and a username
   ending in `bot`.
2. Copy the token it gives you (`123456789:AA...`). Treat it like a password.

Nothing else is needed — no webhook, no public IP, no port forwarding. The bot
uses long polling.

## 2. Install

```bash
git clone https://github.com/GodfatherTHR/minamo.git
cd minamo
```

### Ollama and a model

```bash
ollama pull qwen3:8b          # or any chat model you like
# cloud models need no download, just an ollama.com account:
# ollama run beam:cloud
```

### pi

```bash
curl -fsSL https://pi.dev/install.sh | sh
pi --version
```

Teach pi about your Ollama models. Create or edit `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ollama": {
      "api": "openai-completions",
      "apiKey": "ollama",
      "baseUrl": "http://127.0.0.1:11434/v1",
      "models": [
        { "id": "qwen3:8b", "name": "Qwen3 8B", "reasoning": true,
          "input": ["text"], "contextWindow": 40960, "maxTokens": 4096 }
      ]
    }
  }
}
```

Add one entry per model you want the agent to use. Cloud models work the same
way (`"id": "beam:cloud"`); `contextWindow` can be large, `maxTokens` is the
output cap.

> You don't have to edit this by hand from then on: `/agentmodel <name>` in
> Telegram registers a model with pi automatically.

## 3. Configure

```bash
cp .env.example .env
chmod 600 .env
$EDITOR .env
```

Minimum you must set:

```ini
TELEGRAM_BOT_TOKEN=123456789:AAHe...your-token...
```

Useful defaults to review:

```ini
OLLAMA_MODEL=qwen3:8b          # the chat model
PI_MODEL=qwen3:8b              # the agent model (must satisfy models.json)
WORKSPACE_ROOT=/home/you       # file tools and shell cwd stay inside this
OWNER_CHAT_ID=                 # empty = first /start claims the bot
```

## 4. Run

```bash
python3 bot.py --check    # verifies token, Ollama, pi, model, workspace
./scripts/run.sh          # foreground
```

Open Telegram, message your bot, send `/start`. The first chat to do so becomes
the owner and is locked in; everyone else gets "This is a private bot."

```ini
# Later, to set it explicitly instead of claiming:
OWNER_CHAT_ID=123456789   # /id in the bot tells you the number
```

## Using it

| Command | What it does |
| --- | --- |
| *any text* | ask the chat model |
| `@<task>` | run a coding task with pi, e.g. `@find why tests fail and fix it` |
| `/agentmodel [name]` | show / switch the pi model (auto-registers it) |
| `/auto on\|off` | skip approval prompts for this chat |
| `!<command>` | run a shell command now, e.g. `!git status` |
| `/bg <command>` | background job; you get a message when it exits |
| `/jobs` · `/log <id> [lines]` · `/kill <id>` | manage background jobs |
| `/models` | every Ollama model, with chat/pi markers |
| `/model <name>` | switch the chat model |
| `/cd [path]` · `/ls [path]` · `/cat <path>` | navigate and read |
| `/write <path>` | write a file from the following lines |
| `/new` | clear history and start a fresh pi session |
| `/stop` | stop the current reply or the running pi process |
| `/status` | models, cwd, pi state, jobs, uptime |
| `/think on\|off` | show or hide the chat model's reasoning |

### Example session

```
you:  @create a Makefile with test and lint targets, then run make test
bot:  🤖 pi · beam:cloud · high
      ▶ create a Makefile ...
      🔧 read: Makefile
      ✅ read · 0.2s
      🔐 pi wants to write Makefile (312 bytes)
      [ ✅ Approve ] [ ❌ Deny ]
      ...
      ✅ Done — Makefile now has test and lint targets; make test passes.
      ⏱ 21.4s · beam:cloud · session minamo-1234-1
```

## Models

- `/models` shows what Ollama reports (including cloud models with no local
  weights) plus models configured for pi, and marks which is the chat model and
  which is the agent.
- `/model <anything>` sets the chat model; Ollama pulls or routes it on first use.
- `/agentmodel <anything>` sets the agent model and writes it into
  `~/.pi/agent/models.json` if it isn't there yet.

Cloud models (`name:cloud`) get a larger generation budget automatically, since
their reasoning runs on the provider and costs nothing locally.

## How approvals work

Pi has no built-in permission system. This repo adds one as a pi extension:

- `~/.pi/agent/extensions/minamo-approval.ts` hooks pi's `tool_call` event.
- When launched by the bot, pi gets `MINAMO_APPROVAL_URL`, so every risky call is
  POSTed to `127.0.0.1:8787/approve` and **blocks** until the bot answers.
- `read`, `ls`, `find`, `grep` are allowed silently.
- `bash`, `write`, `edit` send Telegram buttons. Timeout, bridge failure, or a
  non-owner chat id **blocks** the call (fail-safe).
- Run `pi` yourself and the extension is inert — no env var, no gating.

Tool policy is set by `PI_TOOLS` (default `read,ls,find,grep,bash,edit,write`).
A short denylist refuses things like `rm -rf /`, `mkfs`, `dd of=/dev/...`,
`shutdown`, `curl | sh`, `sudo` and force-pushes, even when you approve.

## Background jobs

`/bg` starts the command in its own systemd user unit (`systemd-run --collect`),
so it is outside the bot's cgroup: restarting or upgrading the bot does not kill
your build. Where `systemd-run` is unavailable the bot falls back to `setsid`.

Each job writes `<data>/jobs/<id>.log` and `<id>.exit`. The bot polls those files,
so even after a restart it still reports the exit code and the tail of the log.
Bind a job to a `Ctrl-C`-able stop button with `/kill <id>`.

## Run it as a service

Linux (systemd user unit; auto-restart; starts at boot):

```bash
./scripts/install-service.sh
sudo loginctl enable-linger "$USER"   # only if `loginctl show-user $USER` says Linger=no
```

macOS (launchd agent; starts at login; KeepAlive):

```bash
./scripts/install-launchd.sh
```

Both wrap `scripts/run.sh`, so environment or interpreter changes belong there.

## Configuration reference

| Key | Default | Meaning |
| --- | --- | --- |
| `TELEGRAM_BOT_TOKEN` | — | **required** |
| `OWNER_CHAT_ID` | empty | lock to one chat; empty = first `/start` claims |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `qwen3:8b` | chat model |
| `OLLAMA_NUM_CTX` | `8192` | chat context window |
| `OLLAMA_NUM_PREDICT` | `900` | chat output cap (cloud models get ≥8192) |
| `OLLAMA_KEEP_ALIVE` | `-1` | `-1` keeps the model resident; a duration like `30m` lets it unload |
| `OLLAMA_TIMEOUT` | `600` | seconds for a model request |
| `WORKSPACE_ROOT` | `$HOME` | nothing outside this is readable/writable by tools |
| `SHELL_TIMEOUT` | `60` | inline `!command` timeout; use `/bg` for longer |
| `AGENT_AUTO_APPROVE` | `0` | `1` = run pi's risky tools without asking |
| `PI_BIN` | auto | absolute path to `pi` if it is not on PATH |
| `PI_PROVIDER` | `ollama` | pi provider name |
| `PI_MODEL` | `qwen3:8b` | agent model |
| `PI_THINKING` | `high` | `off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `PI_TOOLS` | `read,ls,find,grep,bash,edit,write` | tools the agent may use |
| `PI_TIMEOUT` | `900` | kill a pi run that exceeds this |
| `PI_APPROVAL_TIMEOUT` | `900` | how long an approval stays pending |
| `APPROVAL_PORT` | `8787` | localhost-only approval bridge |
| `EDIT_INTERVAL` | `1.3` | seconds between Telegram message edits |
| `HISTORY_MESSAGES` / `HISTORY_CHARS` | `16` / `12000` | chat context sent to the model |
| `DATA_DIR` | `./data` | SQLite, logs, job output |
| `DEBUG` | `0` | verbose logging |

## Docker (optional)

Not shipped as a Dockerfile because the useful version of this bot is the one
with access to your real machine. If you want isolation, run the bot and Ollama in
a container and mount only the project directory you want it to touch; see
[pi's containerization guide](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/containerization.md).

## Troubleshooting

**The bot doesn't answer.**
`systemctl --user status minamo-bot` and `journalctl --user -u minamo-bot -f`.
A 401 from Telegram means the token is wrong or was revoked.

**`could not start pi: No such file or directory`.**
The service PATH is minimal. Set `PI_BIN=/home/you/.npm-global/bin/pi` in `.env`
and restart. `python3 bot.py --check` prints the path it resolved.

**pi says the model is unknown.**
It must be in `~/.pi/agent/models.json`. Run `/agentmodel <name>` in Telegram,
or add an entry manually (see step 2).

**Approvals never appear and pi hangs.**
Only the bot starts the bridge. Check `journalctl` for
`approval bridge listening on 127.0.0.1:8787`; the port in `APPROVAL_PORT` must
match. If the bridge is down, pi blocks rather than running unapproved.

**Chat replies "no answer — used its token budget on reasoning".**
A thinking model needs more output tokens. Raise `OLLAMA_NUM_PREDICT`, lower
`PI_THINKING`/`think` level, or switch to a non-thinking model with `/model`.

**Ollama is thrashing between models.**
Two large models cannot both stay resident. Set `OLLAMA_KEEP_ALIVE=30m` so the
idle one unloads, or use cloud models for the agent.

## Development

```bash
python3 -m unittest tests.test_tools tests.test_pi tests.test_bot -v
MINAMO_LIVE_PI=1 python3 -m unittest tests.test_pi.LivePiTests -v   # real model
```

The suite covers the reasoning splitter, message chunking, SQLite store, path
sandbox, command denylist, background jobs, pi's JSON event parsing, the approval
bridge (approve/deny/timeout/auto/foreign chat), and a live run where pi creates a
real file.

Layout:

| File | Responsibility |
| --- | --- |
| `bot.py` | Telegram update loop, commands, chat streaming, job monitor |
| `pi_agent.py` | spawns pi, parses JSON events, approval bridge + pending decisions |
| `llm.py` | Ollama client, reasoning splitter, chunking |
| `tools.py` | workspace sandbox, denylist, background jobs |
| `store.py` | SQLite: history, state, jobs, approvals |
| `telegram_client.py` | minimal Bot API client with retries and rate-limit handling |
| `config.py` | `.env` parsing and defaults |

## Security

The bot has a shell on your machine. Two boundaries keep that sane: the owner
lock (only your chat id) and the approval buttons (only actions you tap through).
`/auto on` removes the second one.

- Keep `.env` at mode `600`; it is gitignored.
- **Rotate the token** if it has ever been pasted into a chat, a log, or a shell
  history: revoke it in @BotFather and put the new one in `.env`.
- Never set `OWNER_CHAT_ID` to a group chat.
- Tools run as the user the bot runs as. Containerize if you need a harder
  boundary.

## License

MIT — see [LICENSE](LICENSE).

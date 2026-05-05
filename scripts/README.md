# PromptChain Scripts (Agent-Authored Workspace)

This is where agent-written PromptChain scripts land for execution and observation. Co-locating runs makes the closed loop work: every script is mlflow-tracked, easy to grep/review, and discoverable by `sio scan`.

## Convention (enforced by `~/.claude/skills/promptchain.md`)

When the agent is asked to **write and run** a PromptChain script (not just explain), it must:

1. Create a per-run subfolder: `scripts/runs/<YYYY-MM-DD>_<short-name>/`
2. Drop `run.py` there. The first lines must be:
   ```python
   from promptchain.observability import init_mlflow
   init_mlflow()
   ```
3. Drop a `README.md` in the same subfolder with:
   - **Intent** — what the script is supposed to do
   - **Expected output** — what success looks like
   - **Triggered by** — link to the conversation/task (e.g., session ID, JIRA, or "ad-hoc")
4. Tell the user how to run it (use `bash scripts/observe.sh runs/<folder>`).

## Running

From the repo root:

```bash
# One-shot run with MLflow tracking enabled
bash scripts/observe.sh runs/2026-05-05_my-script

# Or directly (if you already activated the venv and exported keys)
python scripts/runs/2026-05-05_my-script/run.py
```

## Why commit these?

Audit trail. Every agent-authored script is in git history. When something works (or breaks), you can:
- Reproduce: check out the commit, re-run with `observe.sh`
- Inspect: `git log scripts/runs/` to see what the agent built this week
- Mine: feed run dirs to `sio scan` to find recurring patterns

If a run is truly ephemeral / experimental, drop it in `scripts/scratch/` (gitignored — see `.gitignore`).

## Folder layout

```
scripts/
├── README.md             # this file
├── observe.sh            # run-with-tracking wrapper
├── runs/                 # agent-authored, COMMITTED
│   ├── .gitkeep
│   └── YYYY-MM-DD_<name>/
│       ├── run.py
│       ├── README.md
│       └── output.log    # gitignored
└── scratch/              # ephemeral, GITIGNORED
    └── .gitkeep
```

## Anti-patterns

- ❌ Dropping scripts at repo root (`./test.py`, `./demo.py`) — they get lost.
- ❌ Putting them in `examples/` — that's for canonical, package-shipped examples, not agent experiments.
- ❌ Skipping `init_mlflow()` — kills the observability signal that powers `sio scan`.
- ❌ No README in the run folder — six weeks later, no one remembers what `run.py` was for.

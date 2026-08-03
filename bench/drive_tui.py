#!/usr/bin/env python3
"""Drive the promptchain TUI in a tmux pane to dogfood UI errors.

Replays a JSONL task stream into the running TUI (tmux send-keys), waits for the
screen to settle, then scans the NEW lines of the --dev debug log for tracebacks
/ errors triggered by that task. Produces a report of which prompts broke the UI.

This is the agent-side of the observe-and-fix loop: it finds errors; a human or
coding agent reads the report and fixes them.

Prereqs: the TUI is already running in a tmux pane (see `pc-harness start`),
launched with `--dev` so a debug log exists.

Usage:
    python drive_tui.py --tasks tau_dogfood_25.jsonl --session dogfood \
        --pane pcdog:0.0 --limit 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ERROR_MARKERS = ("Traceback (most recent call last)", "| ERROR |", "| CRITICAL |")


def tmux(*args: str) -> str:
    return subprocess.run(
        ["tmux", *args], capture_output=True, text=True
    ).stdout


def capture(pane: str) -> str:
    return tmux("capture-pane", "-p", "-t", pane)


def send_prompt(pane: str, text: str) -> None:
    # -l sends the text literally (no key interpretation), then Enter submits.
    subprocess.run(["tmux", "send-keys", "-t", pane, "-l", text])
    time.sleep(0.2)
    subprocess.run(["tmux", "send-keys", "-t", pane, "Enter"])


def newest_log(session: str) -> Path | None:
    log_dir = Path.home() / ".promptchain" / "sessions" / session
    logs = sorted(log_dir.glob("debug_*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1] if logs else None


def wait_for_turn(pane: str, min_wait: float, max_wait: float):
    """Wait for one agent turn by tracking the 'Processing…' indicator lifecycle.

    Done when the indicator appeared and then cleared (turn complete) and the
    screen settled, or max_wait. Returns (duration, saw_processing). A flat
    6s/no-processing result means the agent never actually ran.
    """
    start = time.time()
    time.sleep(min_wait)
    saw = False
    stable = 0
    prev = None
    while time.time() - start < max_wait:
        cur = capture(pane)
        processing = ("Processing with" in cur) or ("⏳" in cur)
        if processing:
            saw = True
        stable = stable + 1 if cur == prev else 0
        prev = cur
        # Turn complete: indicator showed and is now gone, screen settled.
        if saw and not processing and stable >= 2:
            break
        # Fallback: agent never showed an indicator but the screen long-settled.
        if not saw and stable >= 4 and (time.time() - start) > 20:
            break
        time.sleep(3.0)
    return time.time() - start, saw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--session", default="dogfood", help="promptchain --session name")
    ap.add_argument("--pane", default="pcdog:0.0", help="tmux target pane of the TUI")
    ap.add_argument("--limit", type=int, default=0, help="cap number of tasks (0=all)")
    ap.add_argument("--min-wait", type=float, default=4.0)
    ap.add_argument("--max-wait", type=float, default=200.0)
    ap.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "dogfood_report.json"
    )
    args = ap.parse_args()

    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[: args.limit]

    log = newest_log(args.session)
    if log is None:
        print(f"!! no debug log under ~/.promptchain/sessions/{args.session}/ — "
              f"is the TUI running with --dev? (pc-harness start)")
        return
    print(f"watching log: {log}")

    results = []
    for n, task in enumerate(tasks, 1):
        offset = log.stat().st_size  # mark log position before the task
        print(f"[{n}/{len(tasks)}] {task['id']} ({task['category']}) ... ", end="", flush=True)
        send_prompt(args.pane, task["send"])
        dur, agent_ran = wait_for_turn(args.pane, args.min_wait, args.max_wait)

        # New log bytes since the marker → scan for error signatures.
        with log.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(offset)
            new_log = fh.read()
        hits = [line for line in new_log.splitlines()
                if any(m in line for m in ERROR_MARKERS)]

        rec = {
            "id": task["id"],
            "category": task["category"],
            "duration_s": round(dur, 1),
            "agent_ran": agent_ran,
            "error_count": len(hits),
            "errors": hits[:20],
            "pane_tail": "\n".join(capture(args.pane).splitlines()[-8:]),
        }
        results.append(rec)
        print(f"{rec['duration_s']}s  agent_ran={agent_ran}  errors={rec['error_count']}")

    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    broke = [r for r in results if r["error_count"]]
    print(f"\n=== dogfood complete: {len(results)} tasks, "
          f"{len(broke)} triggered errors ===")
    for r in broke:
        print(f"  ✗ {r['id']} ({r['category']}) — {r['error_count']} error line(s)")
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()

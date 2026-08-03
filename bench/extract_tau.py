#!/usr/bin/env python3
"""Extract tau-bench task instructions into the agent-bench task schema.

Reads tau-bench's task files via AST (no import/execution of tau_bench, so no
dependency on installing the framework) and emits JSONL rows shaped as:

    {"id", "category", "send", "check", "source"}

These are used as the prompt stream for the TUI dogfood harness. Note: the
instructions reference tau-bench's internal DB/tools, so the agent cannot truly
*complete* them without the full env — for dogfooding we only check that the TUI
does not crash (no_error / no_traceback).

Usage:
    python extract_tau.py                 # full corpus -> bench/tau_tasks.jsonl
    python extract_tau.py --slice 25      # a balanced dogfood slice
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path

# tau-bench clone (read-only). Override with --src, or set TAU_BENCH_ENVS.
# Defaults relative to this repo's parent so no author-specific absolute path
# ships in a public repo.
DEFAULT_SRC = Path(
    os.environ.get(
        "TAU_BENCH_ENVS",
        str(Path(__file__).resolve().parents[2] / "tau-bench" / "tau_bench" / "envs"),
    )
)

# (relative path, category label) — the files that hold Task(...) definitions.
TASK_FILES = [
    ("airline/tasks_test.py", "tau_airline"),
    ("retail/tasks_test.py", "tau_retail"),
    ("retail/tasks_dev.py", "tau_retail_dev"),
]


def extract_instructions(py_file: Path) -> list[str]:
    """Return every ``instruction=...`` string literal in a task file via AST."""
    src = py_file.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_file))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "instruction":
            val = node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                out.append(val.value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "tau_tasks.jsonl",
    )
    ap.add_argument(
        "--slice",
        type=int,
        default=0,
        help="If >0, write a balanced dogfood slice of N tasks instead of the full corpus.",
    )
    args = ap.parse_args()

    rows: list[dict] = []
    for rel, category in TASK_FILES:
        f = args.src / rel
        if not f.exists():
            print(f"  skip (missing): {f}")
            continue
        instructions = extract_instructions(f)
        for i, text in enumerate(instructions):
            rows.append(
                {
                    "id": f"{category}-{i:03d}",
                    "category": category,
                    "send": text,
                    # Dogfood check: the TUI must not crash. Task *success* is
                    # out of scope without the tau-bench environment.
                    "check": {"no_error": True, "no_traceback": True},
                    "source": rel,
                }
            )
        print(f"  {rel}: {len(instructions)} tasks")

    if args.slice > 0:
        # Balanced round-robin across categories so the slice is varied.
        by_cat: dict[str, list[dict]] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        sliced: list[dict] = []
        idx = 0
        while len(sliced) < args.slice and any(
            idx < len(v) for v in by_cat.values()
        ):
            for cat in by_cat:
                if idx < len(by_cat[cat]) and len(sliced) < args.slice:
                    sliced.append(by_cat[cat][idx])
            idx += 1
        rows = sliced

    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(rows)} tasks -> {args.out}")


if __name__ == "__main__":
    main()
